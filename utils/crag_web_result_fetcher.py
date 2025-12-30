"""DO NOT MODIFY THIS FILE.

This file has simple functionality to get the full page content of a web search result.
During evaluation, `page_content` attribute would already exist. It is advised to use
this helper class to fetch the full page content instead of using `requests` directly in
your code. Internet access is disabled during evaluation and using `requests` would fail
the submission.
"""

import os
from hashlib import sha256

import requests

CACHE_DIR = os.getenv(
    "CRAG_WEBSEARCH_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache/crag/", "web_search_results"),
)
os.makedirs(CACHE_DIR, exist_ok=True)


class WebSearchResult:
    def __init__(self, result: dict):
        self.result = result

    def _get_cache_filename(self) -> str:
        return os.path.join(
            CACHE_DIR, sha256(self.result["page_url"].encode()).hexdigest()
        )

    def _page_cache_exists(self) -> bool:
        if os.path.exists(self._get_cache_filename()):
            return True
        return False

    def _fetch_page_content(self) -> str:
        response = requests.get(self.result["page_url"])
        response.raise_for_status()
        content = response.text
        with open(self._get_cache_filename(), "w", encoding="utf-8") as f:
            f.write(content)
        self.result["page_content"] = content
        return content

    def _get_page_content_from_cache(self) -> str:
        with open(self._get_cache_filename(), "r", encoding="utf-8") as f:
            return f.read()

    def get(self, key: str, default: str = None) -> str:
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __getitem__(self, key: str) -> str:
        if key == "page_content":
            if "page_content" in self.result:
                return self.result["page_content"]

            if self._page_cache_exists():
                return self._get_page_content_from_cache()
            else:
                return self._fetch_page_content()
        return self.result[key]

    def __len__(self) -> int:
        return len(self.result)

    def __iter__(self):
        return iter(self.result)

    def __getattr__(self, key: str) -> str:
        return self.__getitem__(key)

    def __repr__(self) -> str:
        return str(self.result)

    def __str__(self) -> str:
        return str(self.result)
