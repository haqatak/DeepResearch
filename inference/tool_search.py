from qwen_agent.tools.base import BaseTool, register_tool
from duckduckgo_search import DDGS
from typing import List, Union, Optional

@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs a web search using DuckDuckGo. Supply a 'query' string to get the top 10 search results."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.ddgs = DDGS()

    def _ddg_search(self, query: str, max_results: int = 10):
        """
        Performs a search using DuckDuckGo and formats the results.
        """
        try:
            results = self.ddgs.text(query, max_results=max_results)
            if not results:
                return f"No results found for '{query}'."

            web_snippets = []
            for i, result in enumerate(results):
                snippet = f"{i + 1}. [{result['title']}]({result['href']})\n{result['body']}"
                web_snippets.append(snippet)

            content = f"Search for '{query}' found {len(web_snippets)} results:\n\n" + "\n\n".join(web_snippets)
            return content
        except Exception as e:
            return f"An error occurred during the search: {e}"

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            query = params
        else:
            try:
                query = params["query"]
            except KeyError:
                return "[Search] Invalid request format: Input must be a JSON object containing a 'query' field"

        if isinstance(query, list):
            # Handle list of queries, though the new description encourages a single query.
            # This provides backward compatibility with the agent's potential behavior.
            responses = []
            for q in query:
                responses.append(self._ddg_search(q))
            return "\n=======\n".join(responses)
        else:
            return self._ddg_search(query)

