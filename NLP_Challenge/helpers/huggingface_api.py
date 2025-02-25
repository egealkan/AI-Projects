# import requests
# import os
# from dotenv import load_dotenv
# from huggingface_hub import InferenceClient
# from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage
# from langchain.chat_models.base import BaseChatModel
# from pydantic import Field, PrivateAttr

# class HuggingFaceAPI(BaseChatModel):
#     api_key: str = Field(..., description="API key for authenticating requests to the Hugging Face API")
#     model_name: str = Field(default="tiiuae/falcon-7b-instruct", description="Model name for Hugging Face API")
#     max_tokens: int = Field(default=1000, description="Maximum number of tokens to generate")
#     temperature: float = Field(default=0.7, description="Sampling temperature")
#     _api_url: str = PrivateAttr(default="https://api-inference.huggingface.co/models/")
#     _client: InferenceClient = PrivateAttr()

#     def __init__(self, **data):
#         super().__init__(**data)
#         load_dotenv()
#         self.api_key = self.api_key or os.getenv("HF_API_KEY")
#         if not self.api_key:
#             raise RuntimeError("HuggingFace API key not found.")
        
#         # Initialize the Inference Client
#         self._client = InferenceClient(api_key=self.api_key)
#         self.model_name = self.model_name

#     def _call(self, messages: list[BaseMessage], stop=None, **kwargs):
#         # Properly format messages for the Hugging Face API
#         formatted_messages = []
#         for msg in messages:
#             if isinstance(msg, HumanMessage):
#                 formatted_messages.append({"role": "user", "content": msg.content})
#             elif isinstance(msg, AIMessage):
#                 formatted_messages.append({"role": "assistant", "content": msg.content})
#             elif isinstance(msg, SystemMessage):
#                 formatted_messages.append({"role": "system", "content": msg.content})
#             else:
#                 raise ValueError(f"Unsupported message type: {type(msg)}")

#         headers = {"Authorization": f"Bearer {self.api_key}"}
#         payload = {
#             "inputs": formatted_messages,
#             "parameters": {"max_new_tokens": kwargs.get("max_tokens", self.max_tokens), "temperature": kwargs.get("temperature", self.temperature)}
#         }

#         response = requests.post(f"{self._api_url}{self.model_name}", headers=headers, json=payload)
#         response.raise_for_status()
#         result = response.json()

#         return AIMessage(content=result[0]["generated_text"])

#     def predict(self, prompt):
#         """
#         Make a call to the Hugging Face Inference API and process the response.
#         """
#         headers = {"Authorization": f"Bearer {self.api_key}"}
#         payload = {
#             "inputs": prompt,
#             "parameters": {"max_new_tokens": self.max_tokens, "temperature": self.temperature}
#         }

#         response = requests.post(f"{self._api_url}{self.model_name}", headers=headers, json=payload)
#         response.raise_for_status()
#         result = response.json()

#         # Handle the response based on its structure
#         if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
#             return result[0]["generated_text"]
#         elif isinstance(result, dict) and "generated_text" in result:
#             return result["generated_text"]
#         else:
#             raise RuntimeError(f"Unexpected response format: {result}")

#     def _generate(self, prompt, stop=None, **kwargs):
#         return self.predict(prompt)

#     @property
#     def _llm_type(self):
#         return "huggingface"







import requests
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage, ChatResult, ChatGeneration
from langchain.chat_models.base import BaseChatModel
from pydantic import Field, PrivateAttr


class HuggingFaceAPI(BaseChatModel):
    api_key: str = Field(..., description="API key for authenticating requests to the Hugging Face API")
    model_name: str = Field(default="tiiuae/falcon-7b-instruct", description="Model name for Hugging Face API")
    max_tokens: int = Field(default=1000, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    _api_url: str = PrivateAttr(default="https://api-inference.huggingface.co/models/")
    _client: InferenceClient = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        load_dotenv()
        self.api_key = self.api_key or os.getenv("HF_API_KEY")
        if not self.api_key:
            raise RuntimeError("HuggingFace API key not found.")

        # Initialize the Inference Client
        self._client = InferenceClient(api_key=self.api_key)
        self.model_name = self.model_name

    def _call(self, messages: list[BaseMessage], context=None, stop=None, **kwargs):
        """
        Call the Hugging Face Inference API with optional context.
        """
        formatted_prompt = self._format_prompt(messages, context)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            },
        }

        response = requests.post(f"{self._api_url}{self.model_name}", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]
        else:
            raise RuntimeError(f"Unexpected response format: {result}")

    def _generate(self, messages: list[BaseMessage], stop=None, **kwargs) -> ChatResult:
        """
        Implement the abstract method _generate using the _call method.
        """
        response_text = self._call(messages, stop=stop, **kwargs)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=response_text))],
        )

    def _format_prompt(self, messages: list[BaseMessage], context=None):
        """
        Format messages into a single string prompt for the Hugging Face API.
        """
        if context:
            context_prompt = f"Context: {context}\n\n"
        else:
            context_prompt = ""

        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_messages.append(f"Assistant: {msg.content}")
            elif isinstance(msg, SystemMessage):
                formatted_messages.append(f"System: {msg.content}")
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")

        return context_prompt + "\n".join(formatted_messages)

    def predict(self, prompt, context=None):
        """
        Generate a prediction from the Hugging Face API with optional context.
        """
        formatted_prompt = f"Context: {context}\n\n{prompt}" if context else prompt
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }

        response = requests.post(f"{self._api_url}{self.model_name}", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]
        else:
            raise RuntimeError(f"Unexpected response format: {result}")

    @property
    def _llm_type(self):
        return "huggingface"


