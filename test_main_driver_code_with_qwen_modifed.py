import os
import json
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
import warnings
from concurrent.futures import ThreadPoolExecutor

# --- Dependency Imports ---
import torch
import ffmpeg
import numpy as np
from PIL import Image
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor as VLMAutoProcessor,
    AutoModelForSpeechSeq2Seq, AutoProcessor as ASRAutoProcessor
)
from huggingface_hub import snapshot_download

# ==============================================================================
# --- PREPROCESSING & HELPER FUNCTIONS ---
# ==============================================================================
def _get_video_duration(video_path: str) -> float:
    try:
        probe = ffmpeg.probe(video_path); return float(next(s for s in probe['streams'] if s['codec_type'] == 'video')['duration'])
    except Exception as e:
        print(f"⚠️ Warning: Could not get video duration. Error: {e}"); return 0.0

def _cut_video_clips_fast(video_path: str, clips_dir: str, video_name: str, clip_duration: int):
    if not os.path.exists(clips_dir): os.makedirs(clips_dir)
    try:
        ffmpeg.input(video_path).output(os.path.join(clips_dir, f'{video_name}_%04d.mp4'), f='segment', segment_time=clip_duration, c='copy', reset_timestamps=1).run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
    except ffmpeg.Error as e:
        print(f"❌ FFmpeg Error during cutting: {e.stderr.decode()}"); raise

def _extract_frames_from_clips(clips_directory: str, fps: int = 1):
    for filename in sorted(os.listdir(clips_directory)):
        if filename.lower().endswith('.mp4'):
            input_path = os.path.join(clips_directory, filename)
            output_pattern = os.path.join(clips_directory, f"{os.path.splitext(filename)[0]}_frame_%04d.jpg")
            try:
                ffmpeg.input(input_path).filter('fps', fps=fps).output(output_pattern, start_number=0).run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            except ffmpeg.Error as e:
                print(f"  ⚠️ Warning: Could not extract frames from {filename}. Error: {e.stderr.decode()}")

def _transcribe_audio_local(audio_path: str, asr_model, asr_processor) -> str:
    try:
        out, _ = ffmpeg.input(audio_path, threads=0).output("pipe:", format="s16le", ac=1, ar=16000).run(capture_stdout=True, capture_stderr=True)
        audio_waveform = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
        inputs = asr_processor(audio_waveform, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(asr_model.device, dtype=torch.float16)
        generated_ids = asr_model.generate(input_features)
        transcription = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription
    except Exception as e:
        print(f"⚠️ Warning: Local audio transcription failed. Error: {e}"); return ""

def _extract_and_transcribe_audio(video_path: str, temp_dir: str, asr_model, asr_processor) -> str:
    print("Step 3/3: Extracting and transcribing audio...")
    audio_path = os.path.join(temp_dir, "temp_audio.mp3")
    try:
        ffmpeg.input(video_path).output(audio_path, acodec='libmp3lame').run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        return _transcribe_audio_local(audio_path, asr_model, asr_processor)
    finally:
        if os.path.exists(audio_path): os.remove(audio_path)

class VideoQAManager:
    def __init__(self, args):
        self.args = args; self.clip_duration = args.clip_duration; self.fps = args.fps; self.clips_dir = None
        self.llm_model_name = args.llm_path; self.vlm_model_name = args.vlm_path; self.asr_model_name = args.whisper_path
        self._initialize_local_models()

    def _download_model_if_needed(self, model_name: str):
        print(f"--- Ensuring model '{model_name}' is downloaded ---")
        try:
            snapshot_download(repo_id=model_name, resume_download=True)
            print(f"✅ Model '{model_name}' is available locally.")
        except Exception as e: print(f"❌ Failed to download model '{model_name}'."); raise e

    def _initialize_local_models(self):
        print("--- Loading all local models. This may take a while... ---")
        self._download_model_if_needed(self.llm_model_name); self._download_model_if_needed(self.vlm_model_name); self._download_model_if_needed(self.asr_model_name)
        print(f"🧠 Loading Planner LLM: {self.llm_model_name}..."); self.llm_model = AutoModelForCausalLM.from_pretrained(self.llm_model_name, torch_dtype=torch.bfloat16, device_map="auto"); self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name); print("✅ Planner LLM loaded.")
        print(f"👁️ Loading VLM: {self.vlm_model_name}..."); self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.vlm_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True); self.vlm_processor = VLMAutoProcessor.from_pretrained(self.vlm_model_name, trust_remote_code=True); print("✅ VLM loaded.")
        print(f"🎤 Loading ASR Model: {self.asr_model_name}..."); self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(self.asr_model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True, device_map="auto"); self.asr_processor = ASRAutoProcessor.from_pretrained(self.asr_model_name); print("✅ ASR Model loaded.")

    def call_local_llm(self, messages: List[Dict]) -> str:
        text = self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.llm_model.device)
        generated_ids = self.llm_model.generate(model_inputs.input_ids, max_new_tokens=2048)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        return self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def call_local_vlm(self, query: str, image_paths: List[str]) -> str:
        print(f"   VLM CALL: Analyzing {len(image_paths)} frames...")
        content = [{"type": "text", "text": query}] + [{"type": "image"} for _ in image_paths]
        messages = [{"role": "user", "content": content}]
        text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        loaded_images = [Image.open(p) for p in image_paths]
        inputs = self.vlm_processor(text=[text], images=loaded_images, padding=True, return_tensors="pt").to(self.vlm_model.device)
        generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return self.vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    def preprocess_media(self, video_path: str):
        video_name = Path(video_path).stem
        self.clips_dir = os.path.join(self.args.dataset_folder, 'clips', str(self.clip_duration), video_name)
        print("\n--- Starting Media Preprocessing ---")
        if not (os.path.isdir(self.clips_dir) and any(f.lower().endswith('.mp4') for f in os.listdir(self.clips_dir))):
            print("Step 1/3: Cutting video..."); _cut_video_clips_fast(video_path, self.clips_dir, video_name, self.clip_duration)
        else: print("Step 1/3: Video clips found. Skipping.")
        if not any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(self.clips_dir)):
            print("Step 2/3: Extracting frames..."); _extract_frames_from_clips(self.clips_dir, fps=self.fps)
        else: print("Step 2/3: Image frames found. Skipping.")

    def get_frames_for_clip(self, clip_path: str) -> List[str]:
        clip_dir = Path(clip_path).parent; clip_basename = Path(clip_path).stem
        return sorted([os.path.join(clip_dir, f) for f in os.listdir(clip_dir) if f.startswith(clip_basename) and f.lower().endswith('.jpg')])

    def get_clip_paths(self) -> List[str]:
        if not self.clips_dir or not os.path.isdir(self.clips_dir): return []
        return sorted([os.path.join(self.clips_dir, f) for f in os.listdir(self.clips_dir) if f.lower().endswith('.mp4')])

    def get_sampled_frame_paths(self, num_frames: int) -> List[str]:
        if not self.clips_dir or not os.path.isdir(self.clips_dir): return []
        all_frames = sorted([f for f in os.listdir(self.clips_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not all_frames: return []
        if len(all_frames) <= num_frames: return [os.path.join(self.clips_dir, f) for f in all_frames]
        indices = np.round(np.linspace(0, len(all_frames) - 1, num_frames)).astype(int)
        return [os.path.join(self.clips_dir, all_frames[i]) for i in indices]

    def run_pipeline(self, question: str, video_path: str) -> Dict:
        """Dispatcher: Chooses the right processing method based on the question."""
        self.preprocess_media(video_path)
        sequential_keywords = ["every 10 second clip", "each clip", "sequentially", "clip by clip"]
        if any(keyword in question.lower() for keyword in sequential_keywords):
            print("\n🔍 Sequential analysis task detected. Starting batch processing...")
            return self.run_sequential_analysis(question)
        else:
            print("\n🤖 General Q&A task detected. Starting agentic loop...")
            return self.run_agentic_qa(question, video_path)

    def run_sequential_analysis(self, question: str) -> Dict:
        """Processes each video clip sequentially to answer the user's question for that interval."""
        all_clip_paths = self.get_clip_paths()
        if not all_clip_paths: return {"answer": "Error: No video clips found to analyze."}
        base_question = re.sub(r'by every 10 second clip|each clip|sequentially|clip by clip', '', question, flags=re.IGNORECASE).strip()
        prompt_template = f"For this 10-second video segment, answer the following: '{base_question}'"
        print(f"Using dynamic prompt for each clip: \"{prompt_template}\"")
        analysis_results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_clip = {executor.submit(self.call_local_vlm, query=prompt_template, image_paths=self.get_frames_for_clip(clip_path)): clip_path for clip_path in all_clip_paths if self.get_frames_for_clip(clip_path)}
            for future in future_to_clip:
                clip_path = future_to_clip[future]
                clip_number_match = re.search(r'_(\d{4})\.mp4', Path(clip_path).name)
                if not clip_number_match: continue
                start_time = int(clip_number_match.group(1)) * self.clip_duration
                try:
                    result_text = future.result()
                    print(f"✅ Analysis for {start_time}-{start_time + self.clip_duration}s: {result_text}")
                    analysis_results[start_time] = result_text
                except Exception as e:
                    print(f"❌ Failed to analyze clip {start_time}-{start_time + self.clip_duration}s: {e}"); analysis_results[start_time] = "Error."
        full_report = "\n".join([f"Time {ts}-{ts+self.clip_duration}s: {desc}" for ts, desc in sorted(analysis_results.items())])
        return {"answer": full_report}

    def run_agentic_qa(self, question: str, video_path: str) -> Dict:
        """Modified agentic loop that forces a conclusion and exits immediately."""
        transcript = _extract_and_transcribe_audio(video_path, self.clips_dir, self.asr_model, self.asr_processor)
        duration = _get_video_duration(video_path)
        transcript_context = f"An audio transcript is available: {transcript[:1000]}..." if transcript else "No audio transcript."
        
        system_prompt = """You are an expert multimedia analyst. Your goal is to answer questions using video frames and the provided transcript.
You have one tool: `video_analyzer`. To use it, respond ONLY with a JSON object: {"tool": "video_analyzer", "query": "Your specific question for the vision model."}
After using the tool and receiving an observation, you MUST synthesize all information and provide a final answer inside <answer> tags. Do not ask for more tools after using one."""
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Video Duration: {duration:.0f}s. {transcript_context}\nQuestion: \"{question}\""}]
        
        for i in range(self.args.max_rounds):
            print(f"\n🔄 --- Agent Round {i + 1}/{self.args.max_rounds} --- 🔄\n")
            
            planner_response = self.call_local_llm(messages)
            print(f"🧠 Planner's Response:\n{planner_response}")
            
            # --- FIX: Immediate check for final answer ---
            if '<answer>' in planner_response:
                final_answer_match = re.search(r'<answer>(.*?)</answer>', planner_response, re.DOTALL)
                if final_answer_match:
                    final_answer = final_answer_match.group(1).strip()
                    print("\n" + "="*50 + "\n🎯 Final Answer Found! Exiting loop.\n" + "="*50)
                    return {"answer": final_answer}
            
            messages.append({"role": "assistant", "content": planner_response})

            try:
                json_match = re.search(r'\{.*\}', planner_response, re.DOTALL)
                if json_match:
                    tool_call = json.loads(json_match.group(0))
                    if tool_call.get("tool") == "video_analyzer":
                        frame_paths = self.get_sampled_frame_paths(num_frames=8)
                        observation = self.call_local_vlm(tool_call.get("query", ""), image_paths=frame_paths) if frame_paths else "Error: No frames found."
                        print(f"📊 Observation from VLM:\n{observation}")
                        
                        messages.append({"role": "user", "content": f"Observation from video_analyzer: {observation}\n\nYou now have sufficient information. Provide a comprehensive final answer in <answer> tags."})
                        continue 
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
            
            messages.append({"role": "user", "content": "Your response was not a valid tool call or final answer. Please try again."})
        
        return {"answer": "Max rounds reached without a conclusive answer."}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully Local Multimedia QA with two processing modes.")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--vlm-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--whisper-path", type=str, default="openai/whisper-base")
    parser.add_argument("--dataset_folder", type=str, default='./data')
    parser.add_argument("--clip_duration", type=int, default=10)
    parser.add_argument("--fps", type=int, default=1, help="Frames to extract per second from each clip.")
    parser.add_argument("--max_rounds", type=int, default=5, help="Max rounds for agentic Q&A mode.")
    args = parser.parse_args()
    
    warnings.warn("--- VRAM & RAM WARNING --- This script loads multiple models locally. A high-end GPU setup (>24GB VRAM) and significant RAM (>32GB) is strongly recommended.")

    try:
        manager = VideoQAManager(args)
        result = manager.run_pipeline(question=args.question, video_path=args.video_path)
        print("\n--- FINAL RESULT ---")
        final_output = {"question": args.question, "answer": result.get("answer")}
        print(json.dumps(final_output, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}"); import traceback; traceback.print_exc()