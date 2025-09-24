import json
import os
import signal
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from qwen_agent.tools.base import BaseTool, register_tool
from .prompt import EXTRACTOR_PROMPT
from openai import OpenAI
import random
from urllib.parse import urlparse, unquote
import time
from transformers import AutoTokenizer
import tiktoken

VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))


@staticmethod
def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

def _playwright_worker(url, queue):
    """
    This function runs in a separate process to avoid dependency conflicts.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=60000)
            html_content = page.content()
            browser.close()
        queue.put(html_content)
    except Exception as e:
        queue.put(f"[visit] Failed to read page with Playwright: {e}")

OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""


@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    name = 'visit'
    description = 'Visit a webpage and return a summary of its content.'
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the webpage to visit."
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for the webpage."
            }
        },
        "required": ["url", "goal"]
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except KeyError:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)

        if isinstance(url, list):
            url = url[0]  # Take the first URL if it's a list

        response = self._get_and_summarize_page(url, goal)
        
        print(f'Summary Length {len(response)}; Summary Content {response}')
        return response.strip()

    def _read_page_playwright(self, url: str) -> str:
        """
        Read webpage content using Playwright in a separate process
        to avoid dependency conflicts.
        """
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=_playwright_worker, args=(url, queue))
        process.start()
        process.join(timeout=120)

        if process.is_alive():
            process.terminate()
            process.join()
            return "[visit] Failed to read page: Process timed out."

        html_content = queue.get()

        if html_content.startswith("[visit] Failed"):
            return html_content

        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text
        text = soup.get_text()
        # Break into lines and remove leading/trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text if text else "[visit] Empty content."

    def _call_llm_summarizer(self, msgs, max_retries=2):
        api_key = os.environ.get("API_KEY")
        url_llm = os.environ.get("API_BASE")
        model_name = os.environ.get("SUMMARY_MODEL_NAME", "")
        # If no API_BASE is specified, it likely means we should use the main ollama endpoint
        if not url_llm:
            url_llm = "http://127.0.0.1:11434/v1"

        client = OpenAI(
            api_key=api_key if api_key else "ollama", # Use a dummy key for ollama
            base_url=url_llm,
        )
        for attempt in range(max_retries):
            try:
                chat_response = client.chat.completions.create(
                    model=model_name if model_name else "huggingface.co/gabriellarson/Tongyi-DeepResearch-30B-A3B-GGUF",
                    messages=msgs,
                    temperature=0.7
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        left = content.find('{')
                        right = content.rfind('}')
                        if left != -1 and right != -1 and left <= right:
                            content = content[left:right+1]
                    return content
            except Exception as e:
                if attempt == (max_retries - 1):
                    return ""
                continue
        return ""

    def _get_and_summarize_page(self, url: str, goal: str) -> str:
        content = self._read_page_playwright(url)

        if content and not content.startswith("[visit] Failed"):
            content = truncate_to_tokens(content, max_tokens=95000)
            messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
            
            raw = self._call_llm_summarizer(messages)
            
            summary_retries = 3
            while len(raw) < 10 and summary_retries >= 0:
                truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                content = content[:truncate_length]
                messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
                raw = self._call_llm_summarizer(messages)
                summary_retries -= 1

            if isinstance(raw, str) and raw:
                raw = raw.replace("```json", "").replace("```", "").strip()
                try:
                    parsed_json = json.loads(raw)
                    useful_information = f"The useful information in {url} for user goal {goal} as follows: \n\n"
                    useful_information += f"Evidence in page: \n{parsed_json.get('evidence', '')}\n\n"
                    useful_information += f"Summary: \n{parsed_json.get('summary', '')}\n\n"
                    return useful_information
                except json.JSONDecodeError:
                    pass # Fall through to the failure case

        # If content failed to be read or summarization failed
        useful_information = f"The useful information in {url} for user goal {goal} as follows: \n\n"
        useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed or processed. Please check the URL.\n\n"
        useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available.\n\n"
        return useful_information

    