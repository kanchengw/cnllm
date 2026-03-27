import re


class OutputCleaner:
    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return text
        
        text = re.sub(r'<think>[\s\S]*?</think>', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()