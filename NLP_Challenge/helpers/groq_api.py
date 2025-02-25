from langchain.chat_models.base import BaseChatModel
from langchain.schema import ChatResult, ChatGeneration, AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import Field, PrivateAttr
import requests
from typing import List, Optional, Any, Dict


class GroqAPI(BaseChatModel):
    api_key: str = Field(..., description="API key for authenticating requests to the Groq API")
    model_name: str = Field(default="mixtral-8x7b-32768", description="Model name for Groq API")
    _api_url: str = PrivateAttr(default="https://api.groq.com/openai/v1/chat/completions")

    def __init__(self, **data):
        super().__init__(**data)
        # Remove the 'groq/' prefix if it exists
        if self.model_name.startswith("groq/"):
            self.model_name = self.model_name[5:]

    def _call(self, messages: list[BaseMessage], stop=None, **kwargs):
        """
        Call the Groq API with a list of messages.
        """
        # Properly format messages for the Groq API
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model_name,  # No 'groq/' prefix needed
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "top_p": kwargs.get("top_p", 1.0),
        }

        if stop:
            payload["stop"] = stop

        response = requests.post(self._api_url, headers=headers, json=payload)

        if response.status_code == 200:
            result_content = response.json()["choices"][0]["message"]["content"]
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=result_content)
                    )
                ],
                llm_output={"token_usage": response.json().get("usage", {})}
            )
        else:
            error_message = response.json().get("error", {}).get("message", "Unknown error")
            raise Exception(f"Groq API Error: {response.status_code} - {error_message}")

    def _generate(self, messages: list[BaseMessage], stop=None, **kwargs) -> ChatResult:
        """
        Generate a response using the _call method.
        """
        return self._call(messages, stop=stop, **kwargs)

    def predict(self, text: str, **kwargs: Any) -> str:
        """
        Generate a prediction for a single text input.
        """
        messages = [HumanMessage(content=text)]
        response = self._generate(messages, **kwargs)
        return response.generations[0].message.content

    @property
    def _llm_type(self) -> str:
        return "groq"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "api_url": self._api_url,
        }




