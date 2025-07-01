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

# --- Dependency Imports ---
# NOTE: vLLM and OpenAI are no longer needed
import dashscope # New dependency
from dashscope.api_v1.protocol import Role # For structured conversation
from transformers import AutoTokenizer, AutoProcessor # Still needed for retriever
from video_utils import (_get_video_duration, _cut_video_clips, extract_subtitles, timestamp_to_clip_path,
                         is_valid_video, is_valid_frame, extract_video_clip, parse_subtitle_time, clip_number_to_clip_path)
                         # image_paths_to_base64 is no longer needed for VLM calls
from retriever import Retrieval_Manager
from prompt import *
from moviepy.video.io.VideoFileClip import VideoFileClip
from concurrent.futures import ThreadPoolExecutor

# --- Environment Configuration ---
os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_DS_ROUND = 20

class VideoQAManager:
    """A QA manager that uses the Dashscope API for all model inference."""

    def __init__(self, args, step_callback: Optional[Callable] = None):
        self.args = args
        self.clip_duration = args.clip_duration
        
        # Determine if subtitles should be used and add the attribute to the args object
        self.use_subtitle = bool(args.subtitle_path)
        self.args.use_subtitle = self.use_subtitle
        
        self.clip_fps = args.clip_fps
        self.step_callback = step_callback or self._default_step_callback

        # Store Dashscope model names
        self.llm_model_name = args.llm_model_name
        self.vlm_model_name = args.vlm_model_name

        self._initialize_components()

        self.messages = []
        self.cur_turn = 0

    def _default_step_callback(self, step_data):
        # This can be used for logging if needed
        pass

    def _initialize_components(self):
        """Initializes components and validates the Dashscope API key."""
        try:
            # 1. Validate Dashscope API Key
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            if not dashscope.api_key:
                raise ValueError("The 'DASHSCOPE_API_KEY' environment variable is not set.")
            print("✅ [SUCCESS] Dashscope API key found.")

            # 2. Initialize retriever (still local)
            self.clip_save_folder = f'{self.args.dataset_folder}/clips/{self.args.clip_duration}/'
            self.args.retriever_type = 'large'
            self.retriever = Retrieval_Manager(args=self.args, clip_save_folder=self.clip_save_folder)
            self.retriever.load_model_to_gpu(0)
            print("✅ [SUCCESS] Retrieval Manager initialized")

        except Exception as e:
            print(f"❌ [ERROR] Failed to initialize components: {e}")
            raise

    def single_text2text_with_callback(self, message: List[Dict]):
        """Performs a text-to-text call using the Dashscope Generation API."""
        print(f"🤔 Calling Dashscope Text LLM ({self.llm_model_name})...")
        for retry in range(3):
            try:
                response = dashscope.Generation.call(
                    model=self.llm_model_name,
                    messages=message,
                    result_format='message'
                )
                if response.status_code == 200:
                    return response.output.choices[0].message.content.strip()
                else:
                    print(f"   [WARNING] API call failed (attempt {retry+1}/3): {response.code} - {response.message}")
            except Exception as e:
                print(f"   [WARNING] Exception during API call (attempt {retry+1}/3): {e}")
            if retry < 2:
                time.sleep(5)
        return "" # Return empty if all retries fail

    def _single_vlm_call(self, query: str, frame_paths: List[str]) -> str:
        """Helper function for a single VLM API call."""
        # Construct the content list with local file paths
        content = []
        for p in frame_paths:
            if os.path.exists(p):
                # Dashscope can directly handle local file paths
                content.append({'image': f'file://{os.path.abspath(p)}'})
        content.append({'text': query})

        if len(content) == 1: # No valid images found
            return "Error: No valid images provided for VLM analysis."

        try:
            response = dashscope.MultiModalConversation.call(
                model=self.vlm_model_name,
                messages=[{'role': Role.USER, 'content': content}]
            )
            if response.status_code == 200:
                return response.output.choices[0].message.content.strip()
            else:
                return f"Error from API: {response.code} - {response.message}"
        except Exception as e:
            return f"Exception during VLM API call: {e}"

    def batch_video2text_server(self, queries: List[str], video_paths: List[List[str]]) -> List[str]:
        """Handles batch VLM calls using a thread pool for concurrency."""
        print(f"🧠 Calling Dashscope VLM ({self.vlm_model_name}) for {len(queries)} items concurrently...")
        results = [None] * len(queries)
        with ThreadPoolExecutor(max_workers=10) as executor: # Adjust max_workers as needed
            future_to_index = {
                executor.submit(self._single_vlm_call, query, frames): i
                for i, (query, frames) in enumerate(zip(queries, video_paths))
            }
            for future in tqdm(future_to_index, desc="Processing VLM calls"):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    results[index] = f"Failed to process query: {e}"
        print("✅ VLM processing complete.")
        return results

    # The following methods can remain largely the same, as they call the abstracted functions above.
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

    def process_tools_with_callback(self, output_text, video_path, duration):
        # This function's logic doesn't need to change as it calls the newly implemented
        # batch_video2text_server, which now uses the Dashscope API.
        tool_result = ''
        output_text_clean = output_text.split('</thinking>')[1] if '</thinking>' in output_text else output_text
        if "<video_reader_question>" in output_text:
            # Placeholder for tool logic, assuming it correctly prepares `queries` and `video_clips_list`
            # For brevity, this part is simplified.
            pattern = r"<video_reader>([^<]+)</video_reader>\s*<video_reader_question>([^<]+)</video_reader_question>"
            matches = re.findall(pattern, output_text_clean)
            if matches:
                queries, video_clips_list = [], [] # You need your logic here to populate these
                # Your logic to convert match_set (e.g., '1;2;3') into a list of frame file paths
                # and to populate `queries` and `video_clips_list`
                # Example placeholder:
                for match_set, query in matches:
                    queries.append(query)
                    # This is where you would call timestamp_to_clip_path or similar
                    # to get the list of frame paths for each query
                    video_clips_list.append([]) # Replace with actual frame paths
                
                ans_li = self.batch_video2text_server(queries, video_clips_list)
                for (match_set, _), ans in zip(matches, ans_li):
                    tool_result += f'Video reader result for {match_set}: {ans}\n'
        return tool_result

    def run_pipeline(self, question: str, video_path: str, subtitle_path: str, max_rounds: int, enable_debug: bool) -> Dict:
        self.messages, self.cur_turn = [], 0
        print("\n" + "="*60 + f"\n🚀 Starting Video QA Pipeline for: {Path(video_path).name}\n" + "="*60 + "\n")
        try:
            self.preprocess_video(video_path)
            with VideoFileClip(video_path) as clip: duration = clip.duration
        except Exception as e:
            if enable_debug: raise
            return {"error": f"Preprocessing failed: {e}"}
        
        initial_prompt = self.build_initial_prompt(question, duration, None)
        self.messages = [{"role": Role.USER, "content": initial_prompt}]
        
        while self.cur_turn < max_rounds:
            print(f"\n🔄 --- Round {self.cur_turn + 1}/{max_rounds} --- 🔄\n")
            response = self.single_text2text_with_callback(self.messages)
            if not response: print("❌ Failed to get model response. Aborting."); break
            
            self.messages.append({"role": Role.ASSISTANT, "content": response})
            print("🧠 Assistant's Reasoning & Actions:\n" + response)
            
            if '<answer>' in response:
                final_answer = self.extract_final_answer(response)
                print("\n" + "="*50 + f"\n🎯 Final Answer Found: {final_answer}\n" + "="*50)
                return {"question": question, "answer": final_answer, "conversation": self.messages}
            
            tool_result = self.process_tools_with_callback(response, video_path, duration)
            if tool_result:
                self.messages.append({"role": Role.USER, "content": tool_result})
            else: # If no tools are called, loop again with a nudge
                self.messages.append({"role": Role.USER, "content": "Please continue. Use a tool if you need more information or provide the final answer."})
            
            self.cur_turn += 1
        return {"question": question, "answer": "Max rounds reached", "conversation": self.messages}

    def build_initial_prompt(self, question, duration, subtitles):
        return f"The video is {duration:.0f} seconds long. The user's question is: {question}. Analyze the video using the available tools and provide a concise answer inside <answer></answer> tags."

    def extract_final_answer(self, text: str) -> str:
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        return match.group(1).strip() if match else "Could not extract answer from model response."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video QA CLI using Dashscope APIs")
    
    # Core arguments
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--question", type=str, required=True, help="Question about the video.")
    
    # Dashscope model arguments
    parser.add_argument("--vlm_model_name", type=str, default="qwen-vl-plus", help="Dashscope VLM model ID.")
    parser.add_argument("--llm_model_name", type=str, default="qwen-long", help="Dashscope text LLM model ID.")
    
    # Configuration
    parser.add_argument("--dataset_folder", type=str, default='./data', help="Folder for clips and data.")
    parser.add_argument("--clip_duration", type=int, default=10, help="Duration of video clips (seconds).")
    parser.add_argument("--clip_fps", type=float, default=1.0, help="FPS for frame sampling.")
    parser.add_argument("--max_rounds", type=int, default=10, help="Max conversation rounds.")
    
    # I/O
    parser.add_argument("--subtitle_path", type=str, default=None, help="Optional path to subtitle file.")
    parser.add_tument("--auto_save", action='store_true', help="Save final results to JSON.")
    parser.add_argument("--enable_debug", action='store_true', help="Enable debug mode for verbose errors.")
    
    args = parser.parse_args()

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