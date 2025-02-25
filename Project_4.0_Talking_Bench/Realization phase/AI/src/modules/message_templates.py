from datetime import datetime
import random
from typing import Dict, List

class MessageTemplates:
    def __init__(self):
        self.welcome_messages = {
            "morning": [
                "Goedemorgen! Ik hoop dat je geniet van de frisse ochtendlucht. {question}",
                "Welkom! Deze ochtend is perfect voor een gesprek. {question}",
                "Hallo! Het is een mooie ochtend om te zitten en te praten. {question}"
            ],
            "afternoon": [
                "Hallo! Even een middagpauze? {question}",
                "Hoi! Bedankt dat je vanmiddag langskomt. {question}",
                "Welkom! Het is een perfecte middag voor een gesprek. {question}"
            ],
            "evening": [
                "Goedenavond! Hopelijk neem je even de tijd om tot rust te komen. {question}",
                "Hallo! De avond is een geweldig moment voor een gesprek. {question}",
                "Hoi! Hopelijk geniet je van de avondsfeer. {question}"
            ]
        }
        
        self.farewell_messages = {
            "short_conversation": [
                "Bedankt voor je bezoek! Nog een fijne dag verder.",
                "Leuk je te ontmoeten! Tot ziens.",
                "Bedankt voor het korte gesprek! Hopelijk tot ziens."
            ],
            "medium_conversation": [
                "Echt genoten van ons gesprek! Fijne dag verder.",
                "Bedankt voor het delen van je gedachten. Kom nog eens langs!",
                "Het was gezellig om met je te praten! Hopelijk vond je het ook leuk."
            ],
            "long_conversation": [
                "Bedankt voor zo'n boeiend gesprek! Nog een fantastische dag verder.",
                "Het was heel gezellig om met je te praten. Hopelijk kunnen we ons gesprek een andere keer voortzetten!",
                "Echt bedankt dat je de tijd hebt genomen om te praten. Fijne dag verder!"
            ]
        }

    def get_time_of_day(self) -> str:
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        else:
            return "evening"

    def get_conversation_length(self, duration: float) -> str:
        if duration < 300:  # 5 minutes
            return "short_conversation"
        elif duration < 900:  # 15 minutes
            return "medium_conversation"
        else:
            return "long_conversation"

    def get_welcome_message(self, first_question: str) -> str:
        time_of_day = self.get_time_of_day()
        templates = self.welcome_messages[time_of_day]
        selected_template = random.choice(templates)
        return selected_template.format(question=first_question)

    def get_farewell_message(self, conversation_duration: float) -> str:
        length_category = self.get_conversation_length(conversation_duration)
        return random.choice(self.farewell_messages[length_category])