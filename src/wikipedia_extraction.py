from pathlib import Path
import re

import requests
from bs4 import BeautifulSoup


ARTICLE = "বাংলাদেশ"
API_URL = "https://bn.wikipedia.org/w/api.php"

OUTPUT_FILE = Path("data/raw/wikipedia/bangladesh.txt")

HEADERS = {
    "User-Agent": "BornoLM/0.1 (educational research project)"
}


def clean_text(text: str) -> str:
    """Clean and normalize extracted Bangla text."""

    # Remove citation markers such as [১], [২], [ক]
    text = re.sub(r"\[[^\]]+\]", "", text)

    # Remove Wikipedia's listen marker
    text = re.sub(r"\(\s*শুনুন\s*\)", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([।,!?:;])", r"\1", text)

    # Remove Wikipedia UI elements such as "(শুনুন ⓘ)"
    text = re.sub(r"\(\s*শুনুন(?:\s*[ⓘi])?\s*\)", "", text)
    
    return text.strip()


def fetch_article() -> str:
    params = {
        "action": "parse",
        "page": ARTICLE,
        "prop": "text",
        "format": "json",
        "formatversion": "2",
    }

    response = requests.get(
        API_URL,
        params=params,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if "parse" not in data:
        raise RuntimeError(
            f"Wikipedia API error: {data}"
        )

    return data["parse"]["text"]


def extract_text(html: str) -> str:

    soup = BeautifulSoup(html, "html.parser")

    # Remove elements that aren't useful for pretraining.
    for selector in [
        "table",
        "style",
        "script",
        "noscript",
        "sup.reference",
        ".mw-editsection",
        ".navbox",
        ".vertical-navbox",
        ".metadata",
        ".ambox",
        ".hatnote",
        ".infobox",
        ".reflist",
        "#toc",
    ]:
        for element in soup.select(selector):
            element.decompose()

    sections = []

    # Preserve section headings.
    for element in soup.find_all(["h2", "h3", "h4", "p"]):

        text = element.get_text(
            separator=" ",
            strip=True,
        )

        text = clean_text(text)

        if not text:
            continue

        # Ignore very short fragments.
        if element.name == "p" and len(text) < 30:
            continue

        sections.append(text)

    return "\n\n".join(sections)


def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    html = fetch_article()

    text = extract_text(html)

    if not text:
        raise RuntimeError(
            "No text was extracted from the Wikipedia article."
        )

    OUTPUT_FILE.write_text(
        text,
        encoding="utf-8",
    )

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Characters: {len(text):,}")
    print(f"Paragraphs: {len(text.split(chr(10) + chr(10))):,}")


if __name__ == "__main__":
    main()
