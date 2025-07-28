import re

def clean_text_for_tts(text: str) -> str:
    """
    Remove formatting, markdown, and special characters for smoother TTS.
    """
    print("cleaning txt for tts\n")
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'/\w+\b', '', text)
    text = re.sub(r'[\[\]\{\}\(\)\|\\/]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1 to \2', text)
    text = re.sub(r'-', ' ', text)
    return text.strip()

def extract_after_keyword(text: str, keyword_list: list[str]) -> str | None:
    """
    Returns the portion of text after the first keyword match.
    """
    text = text.lower()
    for keyword in keyword_list:
        pattern = re.compile(rf"\b{re.escape(keyword.lower())}\b\s+(.*)")
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None
