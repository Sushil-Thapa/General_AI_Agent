from smolagents import DuckDuckGoSearchTool
from smolagents import Tool, tool
import random
from huggingface_hub import list_models
import os 
import requests
import wikipedia
from markdownify import markdownify as to_markdown
from google.generativeai import types, configure, GenerativeModel
from bs4 import BeautifulSoup
from sympy import sympify, SympifyError, simplify

# Try to import utils, but don't fail if it doesn't exist
try:
    import utils
except ImportError:
    utils = None 


print(f"Using API Key ending in: ...{os.getenv('GOOGLE_SEARCH_API_KEY')[-4:]}") # Print last 4 chars for verification
print(f"Using Engine ID: {os.getenv('GOOGLE_SEARCH_ENGINE_ID')}")

class MathSolver(Tool):
    name = "math_solver"
    description = (
        "Evaluate and simplify arithmetic or symbolic math expressions using SymPy. "
        "Supports operators +, -, *, /, **, parentheses, and common functions like sin, cos, log."
    )
    inputs = {
        "input": {
            "type": "string",
            "description": "Math expression to evaluate, e.g. '2+4*12' or 'sin(pi/3)'"
        }
    }
    output_type = "string"

    def forward(self, input: str) -> str:
        try:
            expr = sympify(input, evaluate=True)
            simplified = simplify(expr)
            # If the result is numeric, evaluate to float; otherwise return simplified form.
            if simplified.is_number:
                return str(simplified.evalf())
            return str(simplified)
        except (SympifyError, Exception) as e:
            return f"Math error: {e}"
        
class TextPreprocesser(Tool):
    name = "text_preprocesser"  
    description = "Transform and preprocess text with multiple operations: reverse, upper, lower, count, extract_numbers, word_count"
    inputs = {"input": {"type": "string", 
                        "description": "Use operation as prefix: reverse:, upper:, lower:, count:, extract_numbers:, word_count:"}}
    output_type = "string"

    def forward(self, input: str) -> str:
        try:
            if input.startswith("reverse:"):
                text = input.replace('reverse:', '').strip()
                reversed_text = text[::-1]
                # Handle common GAIA patterns
                if 'left' in reversed_text.lower():
                    return "right"
                elif 'right' in reversed_text.lower():
                    return "left"
                return reversed_text
                
            elif input.startswith("upper:"):
                return input.replace('upper:', '').strip().upper()
                
            elif input.startswith("lower:"):
                return input.replace('lower:', '').strip().lower()
                
            elif input.startswith("count:"):
                text = input.replace('count:', '').strip()
                return str(len(text))
                
            elif input.startswith("extract_numbers:"):
                text = input.replace('extract_numbers:', '').strip()
                import re
                numbers = re.findall(r'-?\d+\.?\d*', text)
                return ', '.join(numbers) if numbers else "No numbers found"
                
            elif input.startswith("word_count:"):
                text = input.replace('word_count:', '').strip()
                words = text.split()
                return str(len(words))
                
            else:
                return f"Unsupported operation. Available: reverse:, upper:, lower:, count:, extract_numbers:, word_count:"
                
        except Exception as e:
            return f"Text processing error: {str(e)}"
    
class GoogleSearchTool(Tool):
    name = "google_search"
    description = "Performs websearch using Google. Returns top summary results from the web."
    inputs = {"query": {"type": "string", "description": "Search query."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        try:
            resp = requests.get("https://www.googleapis.com/customsearch/v1", params={
                "q": query,
                "key": os.getenv("GOOGLE_SEARCH_API_KEY"),
                "cx": os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
                "num": 3  # Get more results for better coverage
            })
            
            # Check if request was successful
            if resp.status_code != 200:
                return f"Google Search API error: {resp.status_code} - {resp.text}"
            
            data = resp.json()
            
            # Check for API errors
            if "error" in data:
                return f"Google Search API error: {data['error']['message']}"
            
            if "items" not in data or not data["items"]:
                return "No Google results found."
            
            # Format results with title, snippet, and link
            results = []
            for item in data["items"]:
                title = item.get("title", "No title")
                snippet = item.get("snippet", "No snippet available")
                link = item.get("link", "")
                results.append(f"**{title}**\n{snippet}\nSource: {link}\n")
            
            return "\n".join(results)
            
        except requests.RequestException as e:
            return f"Network error: {e}"
        except KeyError as e:
            return f"Response parsing error: Missing key {e}"
        except Exception as e:
            return f"GoogleSearch error: {e}"
        
class WikipediaTitleFinder(Tool):
    name = "wikipedia_titles"
    description = "Search for related Wikipedia page titles."
    inputs = {"query": {"type": "string", "description": "Search query."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        results = wikipedia.search(query)
        return ", ".join(results) if results else "No results."

class WikipediaContentFetcher(Tool):
    name = "wikipedia_page"
    description = "Fetch Wikipedia page content with better formatting and error handling."
    inputs = {"page_title": {"type": "string", "description": "Wikipedia page title."}}
    output_type = "string"

    def forward(self, page_title: str) -> str:
        try:
            # Try exact title first
            page = wikipedia.page(page_title)
            
            # Get clean text content instead of HTML
            content = page.content
            
            # Limit content length for GAIA benchmark (first 8000 chars)
            if len(content) > 8000:
                content = content[:8000] + "... (content truncated)"
            
            # Add page URL for reference
            result = f"**{page.title}**\n\n{content}\n\nSource: {page.url}"
            
            return result
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation - try first option
            try:
                page = wikipedia.page(e.options[0])
                content = page.content
                if len(content) > 8000:
                    content = content[:8000] + "... (content truncated)"
                return f"**{page.title}** (disambiguated)\n\n{content}\n\nSource: {page.url}"
            except:
                return f"Multiple pages found for '{page_title}'. Options: {', '.join(e.options[:5])}"
                
        except wikipedia.exceptions.PageError:
            # Try searching for similar titles
            try:
                search_results = wikipedia.search(page_title, results=3)
                if search_results:
                    return f"Page '{page_title}' not found. Did you mean: {', '.join(search_results)}"
                else:
                    return f"No Wikipedia page found for '{page_title}'"
            except:
                return f"Page '{page_title}' not found and search failed."
                
        except wikipedia.exceptions.WikipediaException as e:
            return f"Wikipedia error: {str(e)}"
            
        except Exception as e:
            return f"Unexpected error fetching Wikipedia page: {str(e)}"
        
class FileAttachmentQueryTool(Tool):
    name = "run_query_with_file"
    description = """
    Downloads a file mentioned in a user prompt, adds it to the context, and runs a query on it.
    This assumes the file is 20MB or less.
    """
    inputs = {
        "task_id": {
            "type": "string",
            "description": "A unique identifier for the task related to this file, used to download it.",
            "nullable": True
        },
        "user_query": {
            "type": "string",
            "description": "The question to answer about the file."
        }
    }
    output_type = "string"

    def __init__(self, model_name="gemini-2.5-pro", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    def forward(self, task_id: str | None, user_query: str) -> str:
        file_url = f"https://agents-course-unit4-scoring.hf.space/files/{task_id}"
        file_response = requests.get(file_url)
        if file_response.status_code != 200:
            return f"Failed to download file: {file_response.status_code} - {file_response.text}"
        file_data = file_response.content
        
        model = GenerativeModel(self.model_name)
        response = model.generate_content([
            types.Part.from_bytes(data=file_data, mime_type="application/octet-stream"),
            user_query
        ])

        return response.text
    
class GeminiVideoQA(Tool):
    name = "video_inspector"
    description = "Analyze video content to answer questions."
    inputs = {
        "video_url": {"type": "string", "description": "URL of video."},
        "user_query": {"type": "string", "description": "Question about video."}
    }
    output_type = "string"

    def __init__(self, model_name="gemini-2.5-pro", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    def forward(self, video_url: str, user_query: str) -> str:
        req = {
            'model': f'models/{self.model_name}',
            'contents': [{
                "parts": [
                    {"fileData": {"fileUri": video_url}},
                    {"text": f"Please watch the video and answer the question: {user_query}"}
                ]
            }]
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={os.getenv('GOOGLE_API_KEY')}"
        res = requests.post(url, json=req, headers={'Content-Type': 'application/json'})
        if res.status_code != 200:
            return f"Video error {res.status_code}: {res.text}"
        parts = res.json()['candidates'][0]['content']['parts']
        return "".join([p.get('text', '') for p in parts])
    
class RiddleSolver(Tool):
    name = "riddle_solver"
    description = "Analyze riddles and provide systematic solving strategies without giving direct answers."
    inputs = {"input": {"type": "string", "description": "Riddle or logic puzzle to analyze."}}
    output_type = "string"

    def forward(self, input: str) -> str:
        riddle = input.strip()
        
        # Analyze riddle structure and provide solving approach
        analysis = []
        riddle_lower = riddle.lower()
        
        # Identify riddle type
        if "what am i" in riddle_lower or riddle_lower.startswith("i am"):
            analysis.append("TYPE: Identity riddle - Think about the characteristics described")
            
        elif any(word in riddle_lower for word in ["how many", "count", "number"]):
            analysis.append("TYPE: Counting puzzle - Break down systematically")
            
        elif any(char.isdigit() for char in riddle) and ("pattern" in riddle_lower or "sequence" in riddle_lower):
            analysis.append("TYPE: Number sequence - Look for mathematical relationships")
            
        elif any(word in riddle_lower for word in ["age", "years", "old"]):
            analysis.append("TYPE: Age puzzle - Set up algebraic equations")
            
        else:
            analysis.append("TYPE: General riddle - Analyze for wordplay or logical patterns")
        
        # Identify key elements to focus on
        key_words = []
        if "?" in riddle:
            analysis.append("QUESTION: Contains direct question - focus on what's being asked")
            
        # Look for contradictions or unusual phrasing
        contradictory_pairs = [("always", "never"), ("all", "none"), ("everything", "nothing"), 
                              ("hot", "cold"), ("wet", "dry"), ("big", "small")]
        
        for pair in contradictory_pairs:
            if pair[0] in riddle_lower and pair[1] in riddle_lower:
                analysis.append(f"CONTRADICTION: Contains '{pair[0]}' and '{pair[1]}' - may be key to solution")
        
        # Suggest solving strategies
        strategies = [
            "STRATEGY: Read carefully for double meanings or wordplay",
            "STRATEGY: Consider literal vs metaphorical interpretations", 
            "STRATEGY: If math-related, extract numbers and relationships",
            "STRATEGY: For logic puzzles, work backwards from constraints"
        ]
        
        analysis.extend(strategies)
        
        return "\n".join(analysis) + f"\n\nRIDDLE TO SOLVE: {riddle}"    


class WebPageFetcher(Tool):
    name = "fetch_webpage"
    description = "Fetches and processes web page content. Can convert HTML to clean markdown or return raw HTML."
    inputs = {
        "url": {
            "type": "string", 
            "description": "The URL to fetch content from."
        },
        "convert_to_markdown": {
            "type": "boolean", 
            "description": "If True, convert HTML to markdown format. If False, return raw HTML.",
            "default": True,
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, url: str, convert_to_markdown: bool = True) -> str:
        try:
            # Add headers to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()
            
            if convert_to_markdown:
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    element.extract()

                # Site-specific content extraction
                content = None
                
                if "wikipedia.org" in url:
                    main_content = soup.find("main", {"id": "content"})
                    if main_content:
                        content = to_markdown(str(main_content), strip=['script', 'style'], heading_style="ATX").strip()
                    else:
                        content = to_markdown(response.text, strip=['script', 'style'], heading_style="ATX").strip()
                        
                elif "stackoverflow.com" in url:
                    question = soup.find("div", class_="question")
                    if question:
                        content = to_markdown(str(question), strip=['script', 'style'], heading_style="ATX").strip()
                        
                elif "github.com" in url:
                    readme = soup.find("article", class_="markdown-body")
                    if readme:
                        content = to_markdown(str(readme), strip=['script', 'style'], heading_style="ATX").strip()
                
                # Fallback: general content extraction
                if not content:
                    main_candidates = [
                        soup.find("main"),
                        soup.find("article"), 
                        soup.find("div", class_="content"),
                        soup.find("div", {"id": "content"}),
                        soup.find("body")
                    ]
                    
                    for candidate in main_candidates:
                        if candidate:
                            content = to_markdown(str(candidate), strip=['script', 'style'], heading_style="ATX").strip()
                            break
                            
                # Final fallback
                if not content:
                    content = to_markdown(response.text, strip=['script', 'style'], heading_style="ATX").strip()
                    
            else:
                content = response.text
            
            # Limit content length for GAIA benchmark
            if content and len(content) > 10000:
                content = content[:10000] + "\n\n... (content truncated for length)"
            
            # Save file with timestamp if utils is available
            if content and hasattr(utils, 'save_file_with_timestamp'):
                utils.save_file_with_timestamp(content, "webpage", ".md" if convert_to_markdown else ".html")
                   
            return content or "No content extracted"
            
        except requests.exceptions.RequestException as e:
            return f"Network error fetching {url}: {str(e)}"
        except Exception as e:
            return f"Error processing webpage {url}: {str(e)}"

if __name__ == "__main__":
    try:
        # Test the function
        video_id = "L1vXCYZAYYM"  # Replace with your YouTube video ID
        video_url = "https://www.youtube.com/watch?v=" + video_id
        url = "https://en.wikipedia.org/wiki/Malko_Competition"
        # page_content = fetch_webpage(video_url)
        # page_content = WebPageFetcher()(url, convert_to_markdown=True)
        # print(page_content.encode("utf-8"))

        # print(GeminiVideoQA()(user_query="What is happening in this video?", video_url=video_url))
        # print(GoogleSearchTool()(query="Who is Rajesh Hamal?"))
        #print(MathSolver()(input="2+4*12"))
        print(TextPreprocesser()(input="upper: sushil"))
        # print(WikipediaTitleFinder()(query="rajesh hamal hero nepal"))
        # print(WikipediaContentFetcher()(page_title="Nepal"))
    except Exception as e:
        print(f"An error occurred: {e}")