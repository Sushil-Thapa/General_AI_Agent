import os
import gradio as gr
import requests
import asyncio
import threading
import time
import json
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from smolagents import GradioUI, CodeAgent, HfApiModel, ApiModel, InferenceClientModel, LiteLLMModel, ToolCallingAgent, Tool, DuckDuckGoSearchTool
from agent import JarvisAgent

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
CACHE_FILE = "answers_cache.json"
MAX_WORKERS = 3  # Parallel processing limit
BATCH_SIZE = 5   # Process questions in batches

class AnswerCache:
    """Simple file-based cache for answers"""
    def __init__(self, cache_file: str = CACHE_FILE):
        self.cache_file = cache_file
        self._cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
        return {}
    
    def _save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def get(self, task_id: str) -> Optional[str]:
        return self._cache.get(task_id)
    
    def set(self, task_id: str, answer: str):
        self._cache[task_id] = answer
        self._save_cache()
    
    def clear(self):
        self._cache.clear()
        self._save_cache()

class AgentRunner:
    """Manages agent execution with caching and async processing"""
    def __init__(self):
        self.cache = AnswerCache()
        self.agent = None
        self._progress_callback = None
        
    def set_progress_callback(self, callback):
        self._progress_callback = callback
    
    def _update_progress(self, message: str, progress: float = None):
        if self._progress_callback:
            self._progress_callback(message, progress)
    
    def initialize_agent(self) -> bool:
        """Initialize the agent with error handling"""
        try:
            if self.agent is None:
                self.agent = JarvisAgent()
            return True
        except Exception as e:
            self._update_progress(f"Error initializing agent: {e}")
            return False
    
    def process_question(self, task_id: str, question: str, use_cache: bool = True) -> Tuple[str, str]:
        """Process a single question with caching"""
        try:
            # Check cache first
            if use_cache:
                cached_answer = self.cache.get(task_id)
                if cached_answer:
                    return task_id, cached_answer
            
            # Process with agent
            if not self.agent:
                raise Exception("Agent not initialized")
            
            answer = self.agent(question)
            
            # Cache the result
            if use_cache:
                self.cache.set(task_id, answer)
            
            return task_id, answer
            
        except Exception as e:
            error_msg = f"AGENT ERROR: {e}"
            return task_id, error_msg
    
    def process_questions_parallel(self, questions_data: List[Dict], use_cache: bool = True) -> List[Dict]:
        """Process questions in parallel with progress updates"""
        if not self.initialize_agent():
            return []
        
        total_questions = len(questions_data)
        results = []
        completed = 0
        
        self._update_progress(f"Processing {total_questions} questions in parallel...", 0)
        
        # Process in batches to avoid overwhelming the system
        for batch_start in range(0, total_questions, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_questions)
            batch = questions_data[batch_start:batch_end]
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit batch to executor
                future_to_question = {
                    executor.submit(
                        self.process_question, 
                        item["task_id"], 
                        item["question"], 
                        use_cache
                    ): item for item in batch
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_question):
                    item = future_to_question[future]
                    try:
                        task_id, answer = future.result()
                        results.append({
                            "task_id": task_id,
                            "question": item["question"],
                            "submitted_answer": answer
                        })
                        completed += 1
                        progress = (completed / total_questions) * 100
                        self._update_progress(
                            f"Completed {completed}/{total_questions} questions ({progress:.1f}%)",
                            progress
                        )
                    except Exception as e:
                        completed += 1
                        results.append({
                            "task_id": item["task_id"],
                            "question": item["question"],
                            "submitted_answer": f"PROCESSING ERROR: {e}"
                        })
        
        return results

# Global runner instance
runner = AgentRunner()

def fetch_questions(api_url: str = DEFAULT_API_URL) -> Tuple[bool, List[Dict], str]:
    """Fetch questions from the API"""
    questions_url = f"{api_url}/questions"
    
    try:
        print(f"Fetching questions from: {questions_url}")
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        
        if not questions_data:
            return False, [], "Fetched questions list is empty."
        
        print(f"Fetched {len(questions_data)} questions.")
        return True, questions_data, f"Successfully fetched {len(questions_data)} questions."
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching questions: {e}"
        print(error_msg)
        return False, [], error_msg
    except Exception as e:
        error_msg = f"Unexpected error fetching questions: {e}"
        print(error_msg)
        return False, [], error_msg

def submit_answers(username: str, answers: List[Dict], agent_code: str, api_url: str = DEFAULT_API_URL) -> Tuple[bool, str]:
    """Submit answers to the API"""
    submit_url = f"{api_url}/submit"
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": [{"task_id": item["task_id"], "submitted_answer": item["submitted_answer"]} for item in answers]
    }
    
    try:
        print(f"Submitting {len(answers)} answers to: {submit_url}")
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        return True, final_status
        
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except:
            error_detail += f" Response: {e.response.text[:500]}"
        return False, f"Submission Failed: {error_detail}"
        
    except Exception as e:
        return False, f"Submission Failed: {e}"

# State management for async operations
class AppState:
    def __init__(self):
        self.questions_data = []
        self.processed_results = []
        self.is_processing = False
        self.is_submitting = False

app_state = AppState()

def process_questions_async(progress_callback, use_cache: bool = True):
    """Process questions asynchronously"""
    if not app_state.questions_data:
        progress_callback("No questions loaded. Please fetch questions first.", None)
        return
    
    if app_state.is_processing:
        progress_callback("Already processing questions...", None)
        return
    
    app_state.is_processing = True
    
    def run_processing():
        try:
            runner.set_progress_callback(progress_callback)
            app_state.processed_results = runner.process_questions_parallel(
                app_state.questions_data, 
                use_cache
            )
            progress_callback("✅ All questions processed successfully!", 100)
        except Exception as e:
            progress_callback(f"❌ Error during processing: {e}", None)
        finally:
            app_state.is_processing = False
    
    # Run in separate thread
    thread = threading.Thread(target=run_processing, daemon=True)
    thread.start()

def fetch_questions_action():
    """Fetch questions action"""
    success, questions_data, message = fetch_questions()
    
    if success:
        app_state.questions_data = questions_data
        return message, len(questions_data), gr.update(interactive=True), gr.update(interactive=True)
    else:
        return message, 0, gr.update(interactive=False), gr.update(interactive=False)

def get_cached_count():
    """Get count of cached answers"""
    if not hasattr(runner, 'cache'):
        return 0
    return len(runner.cache._cache)

def clear_cache_action():
    """Clear the answer cache"""
    runner.cache.clear()
    return "Cache cleared successfully!", get_cached_count()

def get_results_table():
    """Get current results as DataFrame"""
    if not app_state.processed_results:
        return pd.DataFrame()
    
    display_results = [
        {
            "Task ID": item["task_id"],
            "Question": item["question"][:100] + "..." if len(item["question"]) > 100 else item["question"],
            "Answer": item["submitted_answer"][:200] + "..." if len(item["submitted_answer"]) > 200 else item["submitted_answer"]
        }
        for item in app_state.processed_results
    ]
    
    return pd.DataFrame(display_results)

def submit_answers_action(profile: gr.OAuthProfile | None):
    """Submit answers action"""
    if not profile:
        return "❌ Please log in to Hugging Face first."
    
    if not app_state.processed_results:
        return "❌ No processed results to submit. Please process questions first."
    
    if app_state.is_submitting:
        return "⏳ Already submitting..."
    
    app_state.is_submitting = True
    
    try:
        username = profile.username
        space_id = os.getenv("SPACE_ID")
        agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "N/A"
        
        success, message = submit_answers(username, app_state.processed_results, agent_code)
        return message
    finally:
        app_state.is_submitting = False

# --- Gradio Interface ---
with gr.Blocks(title="Optimized GAIA Agent Runner") as demo:
    gr.Markdown("# 🚀 Optimized GAIA Agent Runner")
    gr.Markdown("""
    **Enhanced Features:**
    - ⚡ **Parallel Processing**: Questions processed concurrently for faster execution
    - 💾 **Smart Caching**: Answers cached to avoid reprocessing 
    - 📊 **Real-time Progress**: Live updates during processing
    - 🔄 **Async Operations**: Non-blocking UI for better user experience
    - 🛡️ **Error Recovery**: Individual question failures don't stop the entire process
    
    **Instructions:**
    1. Log in to your Hugging Face account
    2. Fetch questions from the server
    3. Process questions (with progress tracking)
    4. Submit your answers
    """)
    
    with gr.Row():
        gr.LoginButton()
    
    with gr.Tab("🔄 Process Questions"):
        with gr.Row():
            with gr.Column(scale=2):
                fetch_btn = gr.Button("📥 Fetch Questions", variant="primary")
                fetch_status = gr.Textbox(label="Fetch Status", interactive=False)
                question_count = gr.Number(label="Questions Loaded", value=0, interactive=False)
            
            with gr.Column(scale=1):
                cache_info = gr.Number(label="Cached Answers", value=get_cached_count(), interactive=False)
                clear_cache_btn = gr.Button("🗑️ Clear Cache", variant="secondary")
        
        with gr.Row():
            with gr.Column():
                use_cache = gr.Checkbox(label="Use Cache", value=True)
                process_btn = gr.Button("⚡ Process Questions", variant="primary", interactive=False)
                
        progress_text = gr.Textbox(label="Progress", interactive=False, lines=2)
        progress_bar = gr.Progress()
        
        results_table = gr.DataFrame(label="📊 Results Preview", wrap=True)
    
    with gr.Tab("📤 Submit Results"):
        with gr.Column():
            submit_btn = gr.Button("🚀 Submit to GAIA", variant="primary", size="lg")
            submit_status = gr.Textbox(label="Submission Status", interactive=False, lines=4)
    
    # Event handlers
    fetch_btn.click(
        fn=fetch_questions_action,
        outputs=[fetch_status, question_count, process_btn, submit_btn]
    )
    
    clear_cache_btn.click(
        fn=clear_cache_action,
        outputs=[fetch_status, cache_info]
    )
    
    def start_processing(use_cache_val):
        if app_state.is_processing:
            return "⏳ Already processing...", gr.update()
        
        def progress_update(message, progress):
            return message, progress
        
        # Start processing
        process_questions_async(progress_update, use_cache_val)
        return "🔄 Started processing questions...", gr.update()
    
    def update_progress():
        """Check processing status and update table"""
        table = get_results_table()
        return table
    
    process_btn.click(
        fn=start_processing,
        inputs=[use_cache],
        outputs=[progress_text, progress_bar]
    ).then(
        fn=update_progress,
        outputs=[results_table],
        every=1  # Update every second
    )
    
    submit_btn.click(
        fn=submit_answers_action,
        outputs=[submit_status]
    )

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 OPTIMIZED GAIA AGENT RUNNER")
    print("="*50)
    
    # Environment info
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")
    
    if space_host:
        print(f"✅ SPACE_HOST: {space_host}")
        print(f"   🌐 Runtime URL: https://{space_host}.hf.space")
    
    if space_id:
        print(f"✅ SPACE_ID: {space_id}")
        print(f"   📁 Repo: https://huggingface.co/spaces/{space_id}")
    
    print(f"💾 Cache file: {CACHE_FILE}")
    print(f"⚡ Max workers: {MAX_WORKERS}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print("="*50 + "\n")
    
    demo.launch(debug=True, share=False)
