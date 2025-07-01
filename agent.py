import os, sys, time
from google.generativeai import types, configure

from smolagents import GradioUI, CodeAgent, HfApiModel, ApiModel, InferenceClientModel, LiteLLMModel, ToolCallingAgent, Tool, DuckDuckGoSearchTool
from prompts import SYSTEM_PROMPT
from tools import *

configure(api_key=os.getenv("GOOGLE_API_KEY"))

class JarvisAgent:
    def __init__(self):
        print("JarvisAgent initialized.")
        model = LiteLLMModel(
            model_id="gemini/gemini-2.5-pro",
            api_key=os.getenv("GEMINI_API_KEY"),
            #max_tokens=2000  # Can be higher due to long context window
        )
        
        self.agent = ToolCallingAgent(
            tools=[
                GoogleSearchTool(), 
                MathSolver(),
                TextPreprocesser(),
                WikipediaTitleFinder(),
                WikipediaContentFetcher(),
                FileAttachmentQueryTool(),
                GeminiVideoQA(),
                RiddleSolver(),
                WebPageFetcher(),
            ], 
            model=model,
            add_base_tools=True,
            max_steps=5  # Limit steps for efficiency
        )
        self.agent.prompt_templates["system_prompt"] = SYSTEM_PROMPT
        
    def evaluate_random_questions(self):
        """Test with GAIA-style questions covering different tool types"""
        print("🧪 Running GAIA benchmark validation tests...")
        
        # Define test cases that match real GAIA scenarios
        test_cases = [
            {
                "name": "Math Calculation",
                "question": "What is 15 * 23 + 47?",
                "expected": "392",
                "tools_used": ["math_solver"]
            },
            {
                "name": "Google Search - Current Info",
                "question": "What is the current population of Tokyo in 2024?",
                "expected": "varies",  # We'll check if it returns a number
                "tools_used": ["google_search"]
            },
            {
                "name": "Wikipedia Search",
                "question": "What year was Albert Einstein born?",
                "expected": "1879",
                "tools_used": ["wikipedia_titles", "wikipedia_page"]
            },
            {
                "name": "Text Processing",
                "question": "Extract numbers from this text: 'The meeting is at 3:30 PM on March 15th, room 204'",
                "expected": "varies",  # We'll check if numbers are extracted
                "tools_used": ["text_preprocesser"]
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"🔍 TEST {i}: {test_case['name']}")
            print(f"{'='*60}")
            print(f"📝 Question: {test_case['question']}")
            print(f"✅ Expected: {test_case['expected']}")
            print(f"🛠️  Expected Tools: {', '.join(test_case['tools_used'])}")
            
            try:
                print(f"\n🤖 Running agent...")
                start_time = time.time()
                agent_answer = self(test_case['question'])
                duration = time.time() - start_time
                
                # Clean answer for comparison
                clean_agent = str(agent_answer).replace('[ANSWER]', '').replace('[/ANSWER]', '').strip()
                
                print(f"\n🎯 Agent Answer: {agent_answer}")
                print(f"🔍 Cleaned Answer: {clean_agent}")
                print(f"⏱️  Duration: {duration:.2f} seconds")
                
                # Evaluate based on test type
                is_correct = self._evaluate_answer(test_case, clean_agent)
                
                print(f"📊 Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
                
                results.append({
                    'test': test_case['name'],
                    'question': test_case['question'][:50] + "...",
                    'expected': test_case['expected'],
                    'actual': clean_agent,
                    'correct': is_correct,
                    'duration': duration
                })
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({
                    'test': test_case['name'],
                    'question': test_case['question'][:50] + "...",
                    'expected': test_case['expected'],
                    'actual': f"ERROR: {str(e)[:100]}",
                    'correct': False,
                    'duration': 0
                })
                import traceback
                traceback.print_exc()
        
        # Summary
        self._print_test_summary(results)
    
    def _evaluate_answer(self, test_case, answer):
        """Evaluate answer based on test case type"""
        if test_case['expected'] == "varies":
            # For dynamic answers, check if we got a reasonable response
            if test_case['name'] == "Google Search - Current Info":
                # Check if answer contains numbers (population)
                import re
                return bool(re.search(r'\d+', answer)) and len(answer) > 3
            elif test_case['name'] == "Text Processing":
                # Check if numbers were extracted
                return any(num in answer for num in ['3', '30', '15', '204'])
        else:
            # Exact match for deterministic answers
            return answer == test_case['expected']
        return False
    
    def _print_test_summary(self, results):
        """Print comprehensive test summary"""
        print(f"\n{'='*60}")
        print(f"📈 GAIA VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        correct_count = sum(1 for r in results if r['correct'])
        total_count = len(results)
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        avg_duration = sum(r['duration'] for r in results) / total_count if total_count > 0 else 0
        
        print(f"✅ Correct: {correct_count}/{total_count}")
        print(f"📊 Accuracy: {accuracy:.1f}%")
        print(f"⏱️  Avg Duration: {avg_duration:.2f} seconds")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for i, result in enumerate(results, 1):
            status = "✅" if result['correct'] else "❌"
            print(f"\n{status} Test {i}: {result['test']}")
            print(f"   Q: {result['question']}")
            print(f"   Expected: {result['expected']}")
            print(f"   Got: {result['actual']}")
            print(f"   Time: {result['duration']:.2f}s")
        
        # GAIA readiness assessment
        print(f"\n🎯 GAIA READINESS ASSESSMENT:")
        if accuracy >= 75:
            print("🟢 READY: Agent shows good performance across test types")
        elif accuracy >= 50:
            print("🟡 PARTIAL: Agent needs refinement for some test types")
        else:
            print("🔴 NOT READY: Agent requires significant improvements")
        
        # Tool-specific feedback
        print(f"\n🔧 TOOL PERFORMANCE:")
        print("   📊 Math Solver: Expected to work reliably")
        print("   🔍 Google Search: Check for current information retrieval")
        print("   📖 Wikipedia: Test knowledge base access")
        print("   ✂️  Text Processing: Validate string manipulation")

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:20]}...")
        answer = self.agent.run(question)
        print(f"Agent returning answer: {answer}")
        return str(answer).strip()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print("Usage: python agent.py [question | dev]")
        print(" - Provide a question to get a GAIA-style answer.")
        print(" - Use 'dev' to evaluate 3 random GAIA questions from gaia_qa.csv.")
        sys.exit(0)

    q = " ".join(args)
    agent = JarvisAgent()
    if q == "dev":
        agent.evaluate_random_questions()
    else:
        print(agent(q))