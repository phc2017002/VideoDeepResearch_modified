import os
import json
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
import warnings
# --- Dependency Imports ---
import torch
import ffmpeg
import numpy as np
from PIL import Image
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration, AutoProcessor as VLMAutoProcessor,
    AutoModelForSpeechSeq2Seq, AutoProcessor as ASRAutoProcessor
)

# ==============================================================================
# --- HELPER UTILITY for Qwen-VL ---
# This function is required by the Qwen-VL model as per the documentation.
# ==============================================================================
def process_vision_info(messages: List[Dict[str, Any]]):
    image_inputs = []
    video_inputs = []

    for message in messages:
        if message["role"] == "user":
            content = message["content"]
            for item in content:
                if item["type"] == "image":
                    image_inputs.append(item["image"])
                elif item["type"] == "video":
                    video_inputs.append(item["video"])
    return image_inputs, video_inputs

# ==============================================================================
# --- SELF-CONTAINED VIDEO & AUDIO PREPROCESSING HELPERS ---
# ==============================================================================
def _get_video_duration(video_path: str) -> float:
    try:
        probe = ffmpeg.probe(video_path)
        return float(next(s for s in probe['streams'] if s['codec_type'] == 'video')['duration'])
    except Exception as e:
        print(f"⚠️ Warning: Could not get video duration. Error: {e}")
        return 0.0

def _cut_video_clips_fast(video_path: str, clips_dir: str, video_name: str, clip_duration: int):
    if not os.path.exists(clips_dir): os.makedirs(clips_dir)
    try:
        ffmpeg.input(video_path).output(os.path.join(clips_dir, f'{video_name}_%04d.mp4'), f='segment', segment_time=clip_duration, c='copy', reset_timestamps=1).run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
    except ffmpeg.Error as e:
        print(f"❌ FFmpeg Error during cutting: {e.stderr.decode()}"); raise

def _extract_frames_from_clips(clips_directory: str, fps: int = 1):
    for filename in os.listdir(clips_directory):
        if filename.lower().endswith('.mp4'):
            input_path = os.path.join(clips_directory, filename)
            output_pattern = os.path.join(clips_directory, f"{os.path.splitext(filename)[0]}_frame_%04d.jpg")
            try:
                ffmpeg.input(input_path).filter('fps', fps=fps).output(output_pattern, start_number=0).run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            except ffmpeg.Error as e:
                print(f"  ⚠️ Warning: Could not extract frames from {filename}. Error: {e.stderr.decode()}")

def _transcribe_audio_local(audio_path: str, asr_model, asr_processor) -> str:
    """Transcribes an audio file using a local Whisper model."""
    try:
        # Load audio file and resample to 16kHz as required by Whisper
        out, _ = (
            ffmpeg.input(audio_path)
            .output("pipe:", format="s16le", ac=1, ar=16000)
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio_waveform = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
        
        # Process and transcribe
        inputs = asr_processor(audio_waveform, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(asr_model.device)
        generated_ids = asr_model.generate(input_features)
        transcription = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print("✅ Audio transcription successful.")
        return transcription
    except Exception as e:
        print(f"⚠️ Warning: Local audio transcription failed. Error: {e}")
        return ""

def _extract_and_transcribe_audio(video_path: str, temp_dir: str, asr_model, asr_processor) -> str:
    """Extracts audio, transcribes it locally, and cleans up."""
    print("Step 3/3: Extracting and transcribing audio...")
    audio_path = os.path.join(temp_dir, "temp_audio.mp3")
    transcript = ""
    try:
        ffmpeg.input(video_path).output(audio_path, acodec='libmp3lame').run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        transcript = _transcribe_audio_local(audio_path, asr_model, asr_processor)
    finally:
        if os.path.exists(audio_path): os.remove(audio_path)
    return transcript

class VideoQAManager:
    """A fully local QA manager using Transformers for all AI tasks."""

    def __init__(self, args):
        self.args = args
        self.clip_duration = args.clip_duration
        self.fps = args.fps
        self.clips_dir = None
        self._initialize_local_models()

    def _initialize_local_models(self):
        """Loads all required models into memory."""
        print("--- Loading all local models. This may take a while and require significant VRAM. ---")
        
        # 1. Load Planner LLM (Qwen3)
        print(f"🧠 Loading Planner LLM: {self.args.llm_path}...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.args.llm_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.args.llm_path, torch_dtype="auto", device_map="auto"
        )
        print("✅ Planner LLM loaded.")

        # 2. Load Vision-Language Model (Qwen2.5-VL)
        print(f"👁️ Loading VLM: {self.args.vlm_path}...")
        self.vlm_processor = VLMAutoProcessor.from_pretrained(self.args.vlm_path, trust_remote_code=True)
        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.args.vlm_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        print("✅ VLM loaded.")

        # 3. Load Audio Transcription Model (Whisper)
        print(f"🎤 Loading ASR Model: {self.args.whisper_path}...")
        self.asr_processor = ASRAutoProcessor.from_pretrained(self.args.whisper_path)
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.args.whisper_path, torch_dtype="auto", low_cpu_mem_usage=True, use_safetensors=True, device_map="auto"
        )
        print("✅ ASR Model loaded.")
        print("--- All local models initialized. ---")

    def call_local_llm(self, messages: List[Dict]) -> str:
        """Calls the local Qwen3 model for reasoning, handling the 'thinking' format."""
        print("  LLM CALL: Engaging local thinking...")
        text = self.llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.llm_model.device)

        generated_ids = self.llm_model.generate(**model_inputs, max_new_tokens=2048)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        try:
            think_token_id = 151668  # '</think>' token ID for Qwen
            index = len(output_ids) - output_ids[::-1].index(think_token_id)
        except ValueError:
            index = 0

        thinking_content = self.llm_tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
        content = self.llm_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

        if thinking_content:
            print("\n" + "=" * 20 + " Thinking Process " + "=" * 20)
            print(thinking_content)
            print("=" * 58)
        
        return content

    def call_local_vlm(self, query: str, image_paths: List[str]) -> str:
        """Calls the local Qwen2.5-VL model with a query and images."""
        print(f"   VLM CALL: Analyzing {len(image_paths)} images locally...")
        
        content = [{"type": "text", "text": query}]
        for img_path in image_paths:
            content.append({"type": "image", "image": img_path})
        
        messages = [{"role": "user", "content": content}]
        
        text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.vlm_processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(self.vlm_model.device)
        
        generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        return output_text

    def preprocess_media(self, video_path: str) -> str:
        """Orchestrates local preprocessing: cutting, frame extraction, and transcription."""
        video_name = Path(video_path).stem
        self.clips_dir = os.path.join(self.args.dataset_folder, 'clips', str(self.clip_duration), video_name)
        
        print("\n--- Starting Media Preprocessing ---")
        clips_exist = os.path.isdir(self.clips_dir) and any(f.lower().endswith('.mp4') for f in os.listdir(self.clips_dir))
        if not clips_exist:
            print("Step 1/3: Cutting video...")
            _cut_video_clips_fast(video_path, self.clips_dir, video_name, self.clip_duration)
        else:
            print("Step 1/3: Video clips found. Skipping.")

        frames_exist = os.path.isdir(self.clips_dir) and any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(self.clips_dir))
        if not frames_exist:
            print("Step 2/3: Extracting frames...")
            _extract_frames_from_clips(self.clips_dir, fps=self.fps)
        else:
            print("Step 2/3: Image frames found. Skipping.")
        
        transcript = _extract_and_transcribe_audio(video_path, self.clips_dir, self.asr_model, self.asr_processor)
        print("--- Media Preprocessing Complete ---\n")
        return transcript

    def get_frame_paths_for_analysis(self, num_frames: int) -> List[str]:
        if not self.clips_dir or not os.path.isdir(self.clips_dir): return []
        all_frames = sorted([f for f in os.listdir(self.clips_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not all_frames: return []
        if len(all_frames) <= num_frames: return [os.path.join(self.clips_dir, f) for f in all_frames]
        indices = [int(i * (len(all_frames) - 1) / (num_frames - 1)) for i in range(num_frames)]
        return [os.path.join(self.clips_dir, all_frames[i]) for i in indices]

    def run_pipeline(self, question: str, video_path: str) -> Dict:
        """Main agentic loop using local models."""
        transcript = self.preprocess_media(video_path)
        duration = _get_video_duration(video_path)
        num_frames_available = len(self.get_frame_paths_for_analysis(99999))

        transcript_context = "No audio transcript is available."
        if transcript:
            transcript_context = f"An audio transcript has been extracted. Snippet:\n---\n{transcript[:1500]}...\n---"

        initial_user_prompt = f"""I have a video ({duration:.0f}s) with {num_frames_available} frames and an audio transcript.
{transcript_context}
My question is: "{question}"
Begin analysis."""

        system_prompt = f"""You are an expert multimedia analysis assistant. Your goal is to answer the user's question using video frames and audio transcripts.
You have a tool `video_analyzer` for visual questions. To use it, respond ONLY with a JSON object:
{{"tool": "video_analyzer", "query": "Your specific question for the vision model."}}
First, analyze the transcript and the user's question. If you need visual details, call the tool. Otherwise, answer directly.
Provide your final answer inside <answer> tags."""
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": initial_user_prompt}]
        
        for i in range(self.args.max_rounds):
            print(f"\n🔄 --- Agent Round {i + 1}/{self.args.max_rounds} --- 🔄\n")
            
            planner_response_content = self.call_local_llm(messages)
            messages.append({"role": "assistant", "content": planner_response_content})
            print(f"🧠 Planner's Response:\n{planner_response_content}")

            if '<answer>' in planner_response_content:
                final_answer = re.search(r'<answer>(.*?)</answer>', planner_response_content, re.DOTALL)
                if final_answer:
                    print("\n" + "="*50 + "\n🎯 Final Answer Found!\n" + "="*50)
                    return {"question": question, "answer": final_answer.group(1).strip(), "conversation": messages, "transcript": transcript}
            
            try:
                json_match = re.search(r'\{.*\}', planner_response_content, re.DOTALL)
                if not json_match:
                    messages.append({"role": "user", "content": "Invalid response. Please use the tool or provide an answer."})
                    continue
                tool_call = json.loads(json_match.group(0))
                if tool_call.get("tool") == "video_analyzer":
                    vlm_query = tool_call.get("query", "Describe what you see.")
                    frame_paths = self.get_frame_paths_for_analysis(num_frames=8)
                    observation = self.call_local_vlm(vlm_query, frame_paths) if frame_paths else "Error: No frames found."
                    print(f"📊 Observation from VLM:\n{observation}")
                    messages.append({"role": "user", "content": f"Observation from video_analyzer: {observation}"})
            except (json.JSONDecodeError, TypeError):
                print("   (No valid tool call detected in response)")
                messages.append({"role": "user", "content": "Your response was not valid. Use the tool or provide an answer."})
        
        return {"question": question, "answer": "Max rounds reached.", "conversation": messages, "transcript": transcript}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully Local Multimedia QA with Local LLMs and Whisper")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    
    # Model Paths
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen2-7B-Instruct", help="Path to local LLM.")
    parser.add_argument("--vlm-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Path to local VLM.")
    parser.add_argument("--whisper-path", type=str, default="openai/whisper-base.en", help="Path to local Whisper model (e.g., openai/whisper-medium).")

    # Preprocessing & Agent Arguments
    parser.add_argument("--dataset_folder", type=str, default='./data')
    parser.add_argument("--clip_duration", type=int, default=10)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max_rounds", type=int, default=5)
    
    args = parser.parse_args()
    
    warnings.warn("--- VRAM & RAM WARNING --- This script loads multiple large models locally. A high-end GPU setup (>48GB VRAM) and significant RAM (>64GB) is strongly recommended.")

    try:
        manager = VideoQAManager(args)
        result = manager.run_pipeline(question=args.question, video_path=args.video_path)
        
        print("\n--- FINAL RESULT ---")
        final_output = {"question": result.get("question"), "answer": result.get("answer")}
        print(json.dumps(final_output, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()