from qwen_agent.tools.base import BaseTool, register_tool
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from typing import Union, Optional
import multiprocessing
import urllib.parse

def _playwright_search_worker(query, queue):
    """
    This function runs in a separate process to avoid dependency conflicts.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://duckduckgo.com/html/?q={encoded_query}"
            page.goto(url, timeout=60000)
            html_content = page.content()
            browser.close()
        queue.put(html_content)
    except Exception as e:
        queue.put(f"[search] Failed to read page with Playwright: {e}")

@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs a web search using DuckDuckGo and returns the top results."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def _playwright_search(self, query: str, max_results: int = 5):
        """
        Performs a search on DuckDuckGo using Playwright in a separate process
        to avoid dependency conflicts.
        """
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=_playwright_search_worker, args=(query, queue))
        process.start()
        process.join(timeout=120)

        if process.is_alive():
            process.terminate()
            process.join()
            return "[search] Failed to read page: Process timed out."

        html_content = queue.get()

        if html_content.startswith("[search] Failed"):
            return html_content

        soup = BeautifulSoup(html_content, 'html.parser')
        results = soup.find_all('div', class_='result')

            if not results:
                return f"No results found for '{query}'."

            web_snippets = []
            for i, result in enumerate(results[:max_results]):
                title_tag = result.find('a', class_='result__a')
                snippet_tag = result.find('a', class_='result__snippet')

                if title_tag and snippet_tag:
                    title = title_tag.get_text(strip=True)
                    link = title_tag['href']
                    snippet_text = snippet_tag.get_text(strip=True)

                    # The link from DDG HTML is a redirect, let's clean it.
                    # e.g., /l/?kh=-1&uddg=https%3A%2F%2Fwww.alibaba.com%2F
                    if link.startswith('/l/'):
                        import urllib.parse
                        parsed_url = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                        if 'uddg' in parsed_url:
                            link = parsed_url['uddg'][0]

                    snippet = f"{i + 1}. [{title}]({link})\n{snippet_text}"
                    web_snippets.append(snippet)

            if not web_snippets:
                return f"No results with snippets found for '{query}'."

            content = f"Search for '{query}' found {len(web_snippets)} results:\n\n" + "\n\n".join(web_snippets)
            return content
        except Exception as e:
            print(f"[Search Tool] An exception occurred: {e}")
            return f"An error occurred during the search with Playwright: {e}"

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            query = params
        else:
            try:
                query = params["query"]
            except KeyError:
                return "[Search] Invalid request format: Input must be a JSON object containing a 'query' field"

        # The agent sometimes sends a list. We'll just take the first item.
        if isinstance(query, list):
            if not query:
                return "[Search] Received an empty list of queries."
            query = query[0]

        return self._playwright_search(query)

