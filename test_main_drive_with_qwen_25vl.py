import os
import json
import re
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Callable, Optional
import time
import argparse
import warnings
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO

# --- Dependency Imports ---
from transformers import AutoTokenizer, AutoProcessor
from video_utils import (_get_video_duration, _cut_video_clips, extract_subtitles, timestamp_to_clip_path,
                         is_valid_video, is_valid_frame, extract_video_clip, parse_subtitle_time, clip_number_to_clip_path,
                         image_paths_to_base64)
from retriever import Retrieval_Manager
from prompt import *
from vllm import LLM, SamplingParams
from openai import OpenAI
from moviepy.video.io.VideoFileClip import VideoFileClip
from concurrent.futures import ThreadPoolExecutor

# --- Environment Configuration ---
os.environ["VLLM_USE_MODELSCOPE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.backends.cuda.matmul.allow_tf32 = True

MAX_DS_ROUND = 20

class VideoQAManager:
    """A flexible QA manager that uses a local VLM and a local OR API-based text LLM."""

    def __init__(self, args, step_callback: Optional[Callable] = None):
        self.args = args
        self.device = f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]}"
        self.clip_duration = args.clip_duration
        
        # --- CORRECTED FIX ---
        # Determine if subtitles should be used and add the attribute to the args object.
        # This ensures Retrieval_Manager can access it without errors.
        self.use_subtitle = bool(args.subtitle_path)
        self.args.use_subtitle = self.use_subtitle  # This line is now correct.
        # --- END OF CORRECTION ---
        
        self.clip_fps = args.clip_fps
        self.step_callback = step_callback or self._default_step_callback

        # Conditional setup for the text LLM
        self.use_api_for_text_llm = bool(os.getenv('API_KEY'))

        if self.use_api_for_text_llm:
            print("✅ API keys found. Using API for text generation.")
            self.ds_model_name = os.getenv('API_MODEL_NAME', 'deepseek-chat')
            self.ds_api_base = os.getenv('API_BASE_URL')
            self.ds_api_keys = [os.getenv('API_KEY')]
        else:
            print("⚠️ API keys not found. Attempting to use local model for text generation.")
            if not args.text_llm_path:
                raise ValueError("API keys are not set. You must provide a local text model via --text_llm_path.")
            self.text_llm_model_name = args.text_llm_path
            print(f"   Local text model to be used: {self.text_llm_model_name}")

        self.vlm_model_name = args.mllm_path
        
        # Now, call _initialize_components AFTER all args have been set
        self._initialize_components()

        self.messages = []
        self.cur_turn = 0
        self.current_data = {}
        self.step_history = []

    def _default_step_callback(self, step_data):
        self.step_history.append(step_data)

    def _initialize_components(self):
        """Initializes all required models and components."""
        try:
            # 1. Initialize retriever (always needed)
            self.clip_save_folder = f'{self.args.dataset_folder}/clips/{self.args.clip_duration}/'
            self.args.retriever_type = 'large'
            self.retriever = Retrieval_Manager(args=self.args, clip_save_folder=self.clip_save_folder)
            self.retriever.load_model_to_gpu(0)
            print("✅ [SUCCESS] Retrieval Manager initialized")

            # 2. Load local VLM (always needed)
            print(f"🧠 Loading local VLM '{self.vlm_model_name}' into memory...")
            self.vlm_model = LLM(
                model=self.vlm_model_name,
                trust_remote_code=True,
                gpu_memory_utilization=0.3,
                tensor_parallel_size=torch.cuda.device_count()
            )
            self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_model_name, trust_remote_code=True)
            self.vlm_tokenizer = self.vlm_processor.tokenizer
            print("✅ [SUCCESS] VLM (Qwen-VL) loaded locally.")

            # 3. Conditionally load local Text LLM
            if not self.use_api_for_text_llm:
                print(f"🧠 Loading local TEXT LLM '{self.text_llm_model_name}' into memory...")
                self.text_llm_model = LLM(
                    model=self.text_llm_model_name,
                    trust_remote_code=True,
                )
                self.text_llm_tokenizer = AutoTokenizer.from_pretrained(self.text_llm_model_name)
                print("✅ [SUCCESS] Local Text LLM loaded.")

        except Exception as e:
            print(f"❌ [ERROR] Failed to initialize components: {e}")
            raise

    def single_text2text_with_callback(self, message):
        """Performs a single text-to-text call using either the API or a local model."""
        self.step_callback({'type': 'llm_call', 'status': 'started'})

        if self.use_api_for_text_llm:
            print("🤔 Calling Text LLM API...")
            llm = OpenAI(base_url=self.ds_api_base, api_key=self.ds_api_keys[0])
            for retry in range(3):
                try:
                    completion = llm.chat.completions.create(model=self.ds_model_name, messages=message, timeout=1800)
                    response = completion.choices[0].message.content.strip()
                    self.step_callback({'type': 'llm_call', 'status': 'completed', 'response_length': len(response)})
                    return response
                except Exception as e:
                    print(f"   [WARNING] API call failed (attempt {retry+1}/3): {e}")
                    if retry < 2: time.sleep(5)
            return ""
        else:
            print("🧠 Running local Text LLM inference...")
            prompt = self.text_llm_tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
            sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1500)
            outputs = self.text_llm_model.generate(prompt, sampling_params)
            response = outputs[0].outputs[0].text.strip()
            self.step_callback({'type': 'llm_call', 'status': 'completed', 'response_length': len(response)})
            return response

    def preprocess_video(self, video_path):
        video_name = Path(video_path).stem
        clips_dir = os.path.join(self.args.dataset_folder, 'clips', str(self.clip_duration), video_name)
        os.makedirs(clips_dir, exist_ok=True)
        try:
            duration = _get_video_duration(video_path)
            print(f"📊 Analyzing video... Duration: {duration:.2f}s")
            print("✂️ Cutting video into clips...")
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                executor.submit(_cut_video_clips, video_path, clips_dir, video_name, duration).result()
            print("✅ Video preprocessing completed!")
        except Exception as e:
            print(f"❌ [ERROR] Video preprocessing failed: {e}")
            raise

    def batch_video2text_server(self, queries: List[str], video_paths: List[List[str]]) -> List[str]:
        print(f"🧠 Running local VLM inference for {len(queries)} items...")
        prompts, multi_modal_data_list = [], []
        for query, frame_paths in zip(queries, video_paths):
            pil_images = [Image.open(p) for p in frame_paths if os.path.exists(p)]
            if not pil_images: continue
            content = [{"type": "text", "text": query}] + [{"type": "image"}] * len(pil_images)
            prompt_text = self.vlm_tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_text)
            image_tensors = self.vlm_processor.image_processor(pil_images, return_tensors='pt')['pixel_values']
            multi_modal_data_list.append({'pixel_values': image_tensors.to(self.device)})
        if not prompts: return ["No valid images found for VLM input."] * len(queries)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=2048)
        outputs = self.vlm_model.generate(prompts, sampling_params, multi_modal_data=multi_modal_data_list)
        return [output.outputs[0].text.strip() for output in outputs]

    def process_tools_with_callback(self, output_text, video_path, duration):
        tool_result = ''
        output_text_clean = output_text.split('</thinking>')[1] if '</thinking>' in output_text else output_text
        if "<video_reader_question>" in output_text:
            pattern = r"<video_reader>([^<]+)</video_reader>\s*<video_reader_question>([^<]+)</video_reader_question>"
            matches = re.findall(pattern, output_text_clean)
            queries, video_clips_list = [], []
            for match_set, query in matches:
                # Placeholder for your logic to get frame paths
                queries.append(query)
                # video_clips_list.append(...)
            if queries:
                ans_li = self.batch_video2text_server(queries, video_clips_list)
                for (match_set, _), ans in zip(matches, ans_li):
                    tool_result += f'Video reader result for {match_set}: {ans}\n'
        return tool_result

    def run_pipeline(self, question: str, video_path: str, subtitle_path: str, max_rounds: int, enable_debug: bool) -> Dict:
        self.messages, self.cur_turn, self.step_history = [], 0, []
        print("\n" + "="*60 + f"\n🚀 Starting Video QA Pipeline for: {Path(video_path).name}\n" + "="*60 + "\n")
        try:
            self.preprocess_video(video_path)
            with VideoFileClip(video_path) as clip: duration = clip.duration
        except Exception as e:
            if enable_debug: raise
            return {"error": f"Preprocessing failed: {e}"}
        initial_prompt = self.build_initial_prompt(question, duration, None)
        self.messages = [{"role": "user", "content": initial_prompt}]
        print("🤖 Starting conversation...")
        while self.cur_turn < max_rounds:
            print(f"\n🔄 --- Round {self.cur_turn + 1}/{max_rounds} --- 🔄\n")
            response = self.single_text2text_with_callback(self.messages)
            if not response: print("❌ Failed to get model response. Aborting."); break
            self.messages.append({"role": "assistant", "content": response})
            print("🧠 Assistant's Reasoning & Actions:\n" + response)
            if '<answer>' in response:
                final_answer = self.extract_final_answer(response)
                print("\n" + "="*50 + f"\n🎯 Final Answer Found: {final_answer}\n" + "="*50)
                return {"question": question, "answer": final_answer, "conversation": self.messages}
            print("\n🛠️ Processing tool calls...")
            tool_result = self.process_tools_with_callback(response, video_path, duration)
            if tool_result:
                print("\n📊 Tool Results:\n" + tool_result)
                self.messages.append({"role": "user", "content": tool_result})
            else:
                self.messages.append({"role": "user", "content": "Invalid output or no tools used. Please think again or provide the final answer."})
            self.cur_turn += 1
        return {"question": question, "answer": "Max rounds reached", "conversation": self.messages}

    def build_initial_prompt(self, question, duration, subtitles):
        return f"The video is {duration:.0f} seconds long. The user's question is: {question}. Analyze the video and provide a concise answer inside <answer></answer> tags."

    def extract_final_answer(self, text: str) -> str:
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        return match.group(1).strip() if match else "Could not extract answer."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flexible Video QA CLI (Local/API)")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--question", type=str, required=True, help="Question about the video.")
    parser.add_argument("--mllm_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Path to local VLM model.")
    parser.add_argument("--text_llm_path", type=str, default=None, help="Path to local text-only LLM (used if API keys are not set). E.g., 'meta-llama/Llama-3-8B-Instruct'")
    
    parser.add_argument("--dataset_folder", type=str, default='./data', help="Folder for clips and data.")
    parser.add_argument("--clip_duration", type=int, default=10, help="Duration of video clips (seconds).")
    parser.add_argument("--clip_fps", type=float, default=1.0, help="FPS for frame sampling.")
    parser.add_argument("--max_rounds", type=int, default=10, help="Max conversation rounds.")
    parser.add_argument("--auto_save", action='store_true', help="Save final results to JSON.")
    parser.add_argument("--enable_debug", action='store_true', help="Enable debug mode.")
    parser.add_argument("--subtitle_path", type=str, default=None, help="Optional path to subtitle file.")
    
    args = parser.parse_args()

    if not os.getenv('API_KEY') and args.text_llm_path:
        warnings.warn("Both VLM and Text LLM will be loaded locally. Ensure you have sufficient VRAM (>24GB recommended).")

    try:
        manager = VideoQAManager(args)
        result = manager.run_pipeline(
            question=args.question, video_path=args.video_path, subtitle_path=args.subtitle_path,
            max_rounds=args.max_rounds, enable_debug=args.enable_debug
        )
        print("\n--- FINAL RESULT ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.auto_save and result:
            filename = f"video_qa_result_{Path(args.video_path).stem}_{int(time.time())}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n📁 Results saved to {filename}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        if args.enable_debug:
            import traceback
            traceback.print_exc()