import os
import json
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List

# --- Dependency Imports ---
# Make sure to install these libraries: pip install dashscope "ffmpeg-python" Pillow
import ffmpeg
import dashscope
from http import HTTPStatus
from dashscope.api_entities.dashscope_response import Role
from dashscope.audio.asr import Transcription

# --- Environment and Constants ---
MAX_AGENT_ROUNDS = 5

# ==============================================================================
# --- SELF-CONTAINED VIDEO & AUDIO PREPROCESSING HELPERS ---
# ==============================================================================

def _get_video_duration(video_path: str) -> float:
    """Gets the duration of a video in seconds using ffmpeg-python."""
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        return float(video_info['duration'])
    except (ffmpeg.Error, StopIteration, KeyError) as e:
        print(f"⚠️  Warning: Could not get video duration for {video_path}. Error: {e}")
        return 0.0

def _cut_video_clips_fast(video_path: str, clips_dir: str, video_name: str, clip_duration: int):
    """Cuts a video into smaller clips using a fast ffmpeg stream copy."""
    if not os.path.exists(clips_dir):
        os.makedirs(clips_dir)
    try:
        (
            ffmpeg
            .input(video_path)
            .output(os.path.join(clips_dir, f'{video_name}_%04d.mp4'),
                    f='segment',
                    segment_time=clip_duration,
                    c='copy',
                    reset_timestamps=1)
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
    except ffmpeg.Error as e:
        print("❌ FFmpeg Error during video cutting:")
        print(e.stderr.decode())
        raise

def _extract_frames_from_clips(clips_directory: str, fps: int = 1):
    """Extracts frames from all .mp4 files in a directory."""
    print("Searching for video clips to extract frames from...")
    for filename in os.listdir(clips_directory):
        if filename.lower().endswith('.mp4'):
            input_path = os.path.join(clips_directory, filename)
            output_pattern = os.path.join(clips_directory, f"{os.path.splitext(filename)[0]}_frame_%04d.jpg")
            
            try:
                (
                    ffmpeg
                    .input(input_path)
                    .filter('fps', fps=fps)
                    .output(output_pattern, start_number=0)
                    .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
                )
            except ffmpeg.Error as e:
                print(f"  ⚠️ Warning: Could not extract frames from {filename}. Error: {e.stderr.decode()}")

# --- NEW FUNCTION FOR AUDIO TRANSCRIPTION ---
def _extract_and_transcribe_audio(video_path: str, temp_dir: str) -> str:
    """Extracts audio from video, transcribes it via Dashscope, and cleans up."""
    print("Step 3/3: Extracting and transcribing audio...")
    audio_path = os.path.join(temp_dir, "temp_audio.mp3")
    
    try:
        # Extract audio using ffmpeg
        ffmpeg.input(video_path).output(
            audio_path,
            acodec='libmp3lame',
            audio_bitrate='128k',
            ac=1 # Mono channel is fine for transcription
        ).run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        
        # Transcribe using Dashscope Paraformer ASR
        transcription = Transcription.call(
            model='paraformer-v1',
            file_path=f'file://{os.path.abspath(audio_path)}',
            api_key=os.getenv('DASHSCOPE_API_KEY')
        )

        if transcription.status_code == HTTPStatus.OK:
            print("✅ Audio transcription successful.")
            return transcription.output['text']
        else:
            print(f"⚠️ Warning: Audio transcription failed. Code: {transcription.code}, Message: {transcription.message}")
            return ""

    except ffmpeg.Error as e:
        print(f"⚠️ Warning: Could not extract audio from video. FFmpeg error: {e.stderr.decode()}")
        return ""
    except Exception as e:
        print(f"⚠️ Warning: An unexpected error occurred during transcription. Error: {e}")
        return ""
    finally:
        # Clean up the temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)


class VideoQAManager:
    """An API-powered QA manager that automates and optimizes video and audio preprocessing."""

    def __init__(self, args):
        self.args = args
        self.clip_duration = args.clip_duration
        self.fps = args.fps
        self.clips_dir = None
        
        dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1/"
        
        self.llm_model_name = args.llm_model_name
        self.vl_model_name = args.vl_model_name
        
        if not os.getenv('DASHSCOPE_API_KEY'):
            raise ValueError("DASHSCOPE_API_KEY environment variable not set. Please export your API key.")
        
        print(f"✅ Dashscope API key found. Using LLM: '{self.llm_model_name}' and VLM: '{self.vl_model_name}'")

    # The API call methods remain unchanged
    def call_qwen_vl_max(self, query: str, image_paths: List[str]) -> str:
        # ... (no changes needed)
        print(f"   VLM CALL: Analyzing {len(image_paths)} images with query: '{query[:50]}...'")
        local_file_urls = [f'file://{os.path.abspath(p)}' for p in image_paths]
        messages = [{'role': 'user', 'content': [{'text': query}, *[{"image": url} for url in local_file_urls]]}]
        try:
            response = dashscope.MultiModalConversation.call(model=self.vl_model_name, messages=messages, api_key=os.getenv('DASHSCOPE_API_KEY'))
            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0].message.content
            else:
                return f"Error: VLM API call failed with code {response.code} - {response.message}"
        except Exception as e:
            return f"Error: An exception occurred during the VLM API call: {e}"

    def call_qwen_max(self, messages: List[Dict]) -> str:
        # ... (no changes needed)
        print("  LLM CALL: Engaging Deep Thinking (streaming)...")
        try:
            response_stream = dashscope.Generation.call(model=self.llm_model_name, messages=messages, result_format="message", enable_thinking=True, stream=True, incremental_output=True, api_key=os.getenv('DASHSCOPE_API_KEY'))
            reasoning_content = ""
            full_content = ""
            for chunk in response_stream:
                if chunk.status_code == HTTPStatus.OK:
                    message = chunk.output.choices[0].message
                    if hasattr(message, 'reasoning_content') and message.reasoning_content is not None:
                        reasoning_content += message.reasoning_content
                    if message.content is not None:
                        full_content += message.content
                else:
                    return f"Error during stream: {chunk.code} - {chunk.message}"
            if reasoning_content:
                print("\n" + "=" * 20 + " Thinking Process " + "=" * 20)
                print(reasoning_content.strip())
                print("=" * 58)
            return full_content
        except Exception as e:
            return f"Error: An exception occurred during the LLM API call: {e}"

    # --- MODIFIED PREPROCESSING METHOD ---
    def preprocess_media(self, video_path: str) -> str:
        """
        INTEGRATED WORKFLOW: Cuts video, extracts frames, and transcribes audio, skipping steps if output already exists.
        Returns the audio transcript.
        """
        video_name = Path(video_path).stem
        self.clips_dir = os.path.join(self.args.dataset_folder, 'clips', str(self.clip_duration), video_name)
        
        print("\n--- Starting Media Preprocessing ---")
        try:
            # Step 1: Cut video into clips
            clips_exist = os.path.isdir(self.clips_dir) and any(f.lower().endswith('.mp4') for f in os.listdir(self.clips_dir))
            if not clips_exist:
                print("Step 1/3: Clips not found. Cutting video into clips...")
                _cut_video_clips_fast(video_path, self.clips_dir, video_name, self.clip_duration)
            else:
                print("Step 1/3: Video clips already exist. Skipping.")

            # Step 2: Extract frames from clips
            frames_exist = os.path.isdir(self.clips_dir) and any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(self.clips_dir))
            if not frames_exist:
                print("Step 2/3: Frames not found. Extracting frames from clips...")
                _extract_frames_from_clips(self.clips_dir, fps=self.fps)
            else:
                print("Step 2/3: Image frames already exist. Skipping.")

            # Step 3: Extract and transcribe audio
            transcript = _extract_and_transcribe_audio(video_path, self.clips_dir)
            
            print("--- Media Preprocessing Complete ---\n")
            return transcript
            
        except Exception as e:
            print(f"❌ [FATAL ERROR] Media preprocessing failed: {e}")
            raise

    def get_frame_paths_for_analysis(self, num_frames: int) -> List[str]:
        # ... (no changes needed)
        if not self.clips_dir or not os.path.isdir(self.clips_dir): return []
        all_frames = sorted([f for f in os.listdir(self.clips_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not all_frames: return []
        if len(all_frames) <= num_frames: return [os.path.join(self.clips_dir, f) for f in all_frames]
        indices = [int(i * (len(all_frames) - 1) / (num_frames - 1)) for i in range(num_frames)]
        return [os.path.join(self.clips_dir, all_frames[i]) for i in indices]

    # --- MODIFIED PIPELINE METHOD ---
    def run_pipeline(self, question: str, video_path: str) -> Dict:
        """The main orchestration loop using an agentic approach with audio and video context."""
        transcript = self.preprocess_media(video_path)
        duration = _get_video_duration(video_path)
        num_frames_available = len(self.get_frame_paths_for_analysis(99999))

        if num_frames_available == 0:
            return {"error": "No frames were extracted from the video. Cannot proceed."}

        # --- CONTEXT INJECTION NOW INCLUDES AUDIO TRANSCRIPT ---
        transcript_context = "No audio transcript is available."
        if transcript:
            # Provide a snippet to avoid overwhelming the context window
            transcript_context = f"An audio transcript has been extracted. Here is the beginning:\n\n---\n{transcript[:1500]}...\n---"

        initial_user_prompt = f"""I have a video that is {duration:.0f} seconds long. I have processed it into {num_frames_available} frames and also extracted the audio transcript.

{transcript_context}

Based on all this available information (both visual and audio), please answer my question.

My question is: "{question}"

Begin your analysis by deciding what information you need first. Use your `video_analyzer` tool for visual questions."""

        system_prompt = f"""You are an expert multimedia analysis assistant. Your goal is to answer the user's question about a video by combining information from both the provided audio transcript and visual analysis of the video frames.
You have access to a tool called `video_analyzer` for visual questions. To use it, respond with ONLY a JSON object:
{{"tool": "video_analyzer", "query": "Your specific question for the vision model."}}

**Your process:**
1.  Analyze the user's question and the initial information (video duration, audio transcript snippet).
2.  Decide if you need visual information from the `video_analyzer` tool or if the audio transcript is sufficient.
3.  If you use the tool, analyze its visual observation in combination with the audio transcript.
4.  Provide a final, comprehensive answer inside <answer> tags once you have enough information.
"""
        messages = [
            {"role": Role.SYSTEM, "content": system_prompt},
            {"role": Role.USER, "content": initial_user_prompt}
        ]
        
        # The rest of the agent loop remains the same
        for i in range(self.args.max_rounds):
            print(f"\n🔄 --- Agent Round {i + 1}/{self.args.max_rounds} --- 🔄\n")
            planner_response_content = self.call_qwen_max(messages)
            messages.append({"role": Role.ASSISTANT, "content": planner_response_content})
            print(f"🧠 Planner's Response:\n{planner_response_content}")
            if '<answer>' in planner_response_content:
                final_answer = re.search(r'<answer>(.*?)</answer>', planner_response_content, re.DOTALL)
                if final_answer:
                    print("\n" + "="*50 + f"\n🎯 Final Answer Found!\n" + "="*50)
                    return {"question": question, "answer": final_answer.group(1).strip(), "conversation": messages, "transcript": transcript}
            try:
                json_match = re.search(r'\{.*\}', planner_response_content, re.DOTALL)
                if not json_match:
                    messages.append({"role": Role.USER, "content": "That was not a valid tool call. Please try again or provide a final answer."})
                    continue
                tool_call = json.loads(json_match.group(0))
                if tool_call.get("tool") == "video_analyzer":
                    vlm_query = tool_call.get("query", "Describe what you see.")
                    frame_paths = self.get_frame_paths_for_analysis(num_frames=8)
                    if not frame_paths:
                        observation = "Error: No frames were found to analyze."
                    else:
                        observation = self.call_qwen_vl_max(vlm_query, frame_paths)
                    print(f"📊 Observation from VLM:\n{observation}")
                    user_observation_prompt = f"Observation from video_analyzer: {observation}"
                    messages.append({"role": Role.USER, "content": user_observation_prompt})
            except (json.JSONDecodeError, TypeError):
                print("   (No valid tool call detected in response)")
                messages.append({"role": Role.USER, "content": "Your response was not a valid JSON tool call. Please try again or provide a final answer."})
                pass
        
        return {"question": question, "answer": "Max rounds reached without a final answer.", "conversation": messages, "transcript": transcript}


if __name__ == "__main__":
    # The parser arguments remain the same
    parser = argparse.ArgumentParser(description="Fully Automated Multimedia QA with Dashscope APIs")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--question", type=str, required=True, help="Question about the video.")
    parser.add_argument("--llm_model_name", type=str, default="qwen-plus", help="Dashscope model for reasoning.")
    parser.add_argument("--vl_model_name", type=str, default="qwen-vl-plus", help="Dashscope model for vision analysis.")
    parser.add_argument("--dataset_folder", type=str, default='./data', help="Folder for preprocessed media.")
    parser.add_argument("--clip_duration", type=int, default=10, help="Duration of video clips in seconds.")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract from clips.")
    parser.add_argument("--max_rounds", type=int, default=MAX_AGENT_ROUNDS, help="Max conversation rounds for the agent.")
    
    args = parser.parse_args()

    try:
        manager = VideoQAManager(args)
        result = manager.run_pipeline(question=args.question, video_path=args.video_path)
        
        print("\n--- FINAL RESULT ---")
        # We don't print the full transcript in the final JSON, just the answer.
        # But you could add 'transcript': result.get('transcript', '') if you wanted it.
        final_output = {
            "question": result.get("question"),
            "answer": result.get("answer"),
            "conversation_history": result.get("conversation")
        }
        print(json.dumps(final_output, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()