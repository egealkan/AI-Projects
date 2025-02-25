import random
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class FillerContent:
    thinking_phrases: List[str]
    fun_facts: Dict[str, List[str]]
    engagement_phrases: List[str]

class FillerAgent:
    def __init__(self, bench_data):
        self.fillers = bench_data['location'].get('fillers', [])
        self.thinking_phrases = [
            filler.get("content", "Laat me daar even over nadenken...") 
            for filler in self.fillers
        ] if self.fillers else ["Laat me daar even over nadenken..."]

    def get_filler_content(self, category: Optional[str] = None) -> Dict[str, str]:
        thinking = random.choice(self.thinking_phrases)
        return {
            "thinking": thinking,
            "engagement": thinking
        }

    def get_thinking_phrase(self) -> str:
        return random.choice(self.thinking_phrases)

    def get_engagement_prompt(self) -> str:
        """Een betrokkenheidsprompt ophalen"""
        return random.choice(self.content.engagement_phrases)