from typing import List

from src.copilot.subgraphs.kommo.models import QuotedAnswer
import re

def format_answer(qa: QuotedAnswer, links: List[str]) -> str:
    """Форматирует QuotedAnswer для отображения в чате."""
    parts = []
    seen_urls = set()  # Множество для отслеживания использованных URL изображений

    for idx, citation in enumerate(qa.answer, start=1):
        text = f"{idx}. {citation.quote}"
        existing_images = re.findall(r'!\[\]\((.*?)\)', citation.quote)
        if citation.image_url and citation.image_url not in seen_urls and citation.image_url not in existing_images:
            text += f"\n![]({citation.image_url})"
            seen_urls.add(citation.image_url)
        
        parts.append(text)

    response = "\n\n".join(parts)

    if links:
        links_str = '\n'.join(
            [f'- [{link}]({link}/?utm_content=ai_copilot)'
            for link in links]
        )
        return response + '\n\n' + links_str
    else:
        return response