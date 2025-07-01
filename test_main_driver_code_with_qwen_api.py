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

# --- Environment and Constants ---
MAX_AGENT_ROUNDS = 5

# ==============================================================================
# --- SELF-CONTAINED VIDEO PREPROCESSING HELPERS ---
# ==============================================================================

def _get_video_duration(video_path: str) -> float:
    """Gets the duration of a video in seconds using ffmpeg-python."""
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        return float(video_info['duration'])
    except (ffmpeg.Error, StopIteration, KeyError) as e:
        print(f"⚠️  Warning: Could not get video duration for {video_path}. Defaulting to 0. Error: {e}")
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
    print(f"Searching for video clips in: {clips_directory}")
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


class VideoQAManager:
    """An API-powered QA manager that automates and optimizes video preprocessing."""

    def __init__(self, args):
        self.args = args
        self.clip_duration = args.clip_duration
        self.fps = args.fps
        self.clips_dir = None
        
        # Set the international API endpoint
        dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1/"
        
        # API Configuration
        self.llm_model_name = args.llm_model_name
        self.vl_model_name = args.vl_model_name
        
        if not os.getenv('DASHSCOPE_API_KEY'):
            raise ValueError("DASHSCOPE_API_KEY environment variable not set. Please export your API key.")
        
        print(f"✅ Dashscope API key found. Using LLM: '{self.llm_model_name}' and VLM: '{self.vl_model_name}'")

    def call_qwen_vl_max(self, query: str, image_paths: List[str]) -> str:
        """Calls qwen-vl-max with a query and local image paths."""
        print(f"   VLM CALL: Analyzing {len(image_paths)} images with query: '{query[:50]}...'")
        local_file_urls = [f'file://{os.path.abspath(p)}' for p in image_paths]
        messages = [{'role': 'user', 'content': [{'text': query}, *[{"image": url} for url in local_file_urls]]}]
        
        try:
            response = dashscope.MultiModalConversation.call(
                model=self.vl_model_name,
                messages=messages,
                api_key=os.getenv('DASHSCOPE_API_KEY')
            )
            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0].message.content
            else:
                return f"Error: VLM API call failed with code {response.code} - {response.message}"
        except Exception as e:
            return f"Error: An exception occurred during the VLM API call: {e}"

    def call_qwen_max(self, messages: List[Dict]) -> str:
        """Calls the qwen-plus model, correctly handling the streaming response required by 'enable_thinking'."""
        print("  LLM CALL: Engaging Deep Thinking (streaming)...")
        try:
            response_stream = dashscope.Generation.call(
                model=self.llm_model_name,
                messages=messages,
                result_format="message",
                enable_thinking=True,
                stream=True,
                incremental_output=True,
                api_key=os.getenv('DASHSCOPE_API_KEY')
            )
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

    def preprocess_video(self, video_path: str):
        """INTEGRATED WORKFLOW: Cuts video and extracts frames, skipping steps if output already exists."""
        video_name = Path(video_path).stem
        self.clips_dir = os.path.join(self.args.dataset_folder, 'clips', str(self.clip_duration), video_name)
        
        print("\n--- Starting Video Preprocessing ---")
        try:
            clips_exist = os.path.isdir(self.clips_dir) and any(f.lower().endswith('.mp4') for f in os.listdir(self.clips_dir))
            if not clips_exist:
                print("Step 1/2: Clips not found. Cutting video into clips...")
                _cut_video_clips_fast(video_path, self.clips_dir, video_name, self.clip_duration)
            else:
                print("Step 1/2: Video clips already exist. Skipping cutting step.")

            frames_exist = os.path.isdir(self.clips_dir) and any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(self.clips_dir))
            if not frames_exist:
                print("Step 2/2: Frames not found. Extracting frames from clips...")
                _extract_frames_from_clips(self.clips_dir, fps=self.fps)
            else:
                print("Step 2/2: Image frames already exist. Skipping extraction step.")
            print("--- Video Preprocessing Complete ---\n")
        except Exception as e:
            print(f"❌ [FATAL ERROR] Video preprocessing failed: {e}")
            raise

    def get_frame_paths_for_analysis(self, num_frames: int) -> List[str]:
        """Samples frames evenly across all extracted frames for API analysis."""
        if not self.clips_dir or not os.path.isdir(self.clips_dir):
            return []
        all_frames = sorted([f for f in os.listdir(self.clips_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not all_frames: return []
        if len(all_frames) <= num_frames:
            return [os.path.join(self.clips_dir, f) for f in all_frames]
        indices = [int(i * (len(all_frames) - 1) / (num_frames - 1)) for i in range(num_frames)]
        return [os.path.join(self.clips_dir, all_frames[i]) for i in indices]

    def run_pipeline(self, question: str, video_path: str) -> Dict:
        """The main orchestration loop using an agentic approach."""
        self.preprocess_video(video_path)
        duration = _get_video_duration(video_path)
        num_frames_available = len(self.get_frame_paths_for_analysis(99999))

        if num_frames_available == 0:
            return {"error": "No frames were extracted from the video. Cannot proceed."}

        initial_user_prompt = f"""I have a video that is {duration:.0f} seconds long. I have already processed it and extracted {num_frames_available} frames for you to analyze.

My question is: "{question}"

Please begin your analysis by deciding what to look for first."""

        system_prompt = f"""You are an expert video analysis assistant. Your goal is to answer the user's question about a video.
You have access to a tool called `video_analyzer`. This tool can look at frames from the video and answer specific questions about what is happening visually.
To use the tool, you MUST respond with ONLY a JSON object in the following format:
{{"tool": "video_analyzer", "query": "Your specific question for the vision model. Be very specific."}}

**Your process:**
1.  Analyze the user's initial question and the provided context about the video.
2.  If you need visual information, formulate a precise question for the `video_analyzer` tool and respond with the required JSON. Do not add any other text.
3.  After you receive the information from the tool, analyze it.
4.  If you have enough information, provide the final answer inside <answer> tags.
5.  If you need more visual information, call the `video_analyzer` tool again.
"""
        messages = [
            {"role": Role.SYSTEM, "content": system_prompt},
            {"role": Role.USER, "content": initial_user_prompt}
        ]
        
        for i in range(self.args.max_rounds):
            print(f"\n🔄 --- Agent Round {i + 1}/{self.args.max_rounds} --- 🔄\n")
            
            planner_response_content = self.call_qwen_max(messages)
            messages.append({"role": Role.ASSISTANT, "content": planner_response_content})
            print(f"🧠 Planner's Response:\n{planner_response_content}")

            if '<answer>' in planner_response_content:
                final_answer = re.search(r'<answer>(.*?)</answer>', planner_response_content, re.DOTALL)
                if final_answer:
                    print("\n" + "="*50 + f"\n🎯 Final Answer Found!\n" + "="*50)
                    return {"question": question, "answer": final_answer.group(1).strip(), "conversation": messages}
            
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
                    
                    # --- FINAL FIX IS HERE ---
                    # Feed the observation back as a user message, not a tool message.
                    user_observation_prompt = f"Observation from video_analyzer: {observation}"
                    messages.append({"role": Role.USER, "content": user_observation_prompt})
                    # --- END OF FIX ---

            except (json.JSONDecodeError, TypeError):
                print("   (No valid tool call detected in response)")
                messages.append({"role": Role.USER, "content": "Your response was not a valid JSON tool call. Please try again or provide a final answer."})
                pass
        
        return {"question": question, "answer": "Max rounds reached without a final answer.", "conversation": messages}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully Automated Video QA with Dashscope APIs")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--question", type=str, required=True, help="Question about the video.")
    
    # API Model Arguments
    parser.add_argument("--llm_model_name", type=str, default="qwen-plus", help="Dashscope model for reasoning (e.g., qwen-plus, qwen-max).")
    parser.add_argument("--vl_model_name", type=str, default="qwen-vl-plus", help="Dashscope model for vision analysis.")
    
    # Preprocessing Arguments
    parser.add_argument("--dataset_folder", type=str, default='./data', help="Folder to store preprocessed clips and frames.")
    parser.add_argument("--clip_duration", type=int, default=10, help="Duration of video clips in seconds.")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract from clips.")
    
    # Agent Arguments
    parser.add_argument("--max_rounds", type=int, default=MAX_AGENT_ROUNDS, help="Max conversation rounds for the agent.")
    
    args = parser.parse_args()

    try:
        manager = VideoQAManager(args)
        result = manager.run_pipeline(question=args.question, video_path=args.video_path)
        
        print("\n--- FINAL RESULT ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()