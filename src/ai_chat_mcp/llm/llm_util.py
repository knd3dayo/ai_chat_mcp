from dotenv import load_dotenv
import os, json
import base64
from mimetypes import guess_type
from typing import Any, Union, ClassVar
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Any, Tuple, List
from openai import RateLimitError
import time

import ai_chat_mcp.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class ChatContentItem(BaseModel):
    type: str = Field(default="text", description="The type of content (e.g., 'text', 'image_url').")
    text: Optional[str] = Field(default=None, description="The text content, if type is 'text'.")
    image_url: Optional[dict] = Field(default=None, description="The image URL content, if type is 'image_url'.")

class ChatMessageItem(BaseModel):
    role: str = Field(default="user", description="The role of the message sender (e.g., 'user', 'assistant').")
    content: list[ChatContentItem] = Field(default=[], description="The content of the message, which can be text or other types.")

class CompletionRequest(BaseModel):

    user_role_name: ClassVar[str]  = "user"
    assistant_role_name: ClassVar[str]  = "assistant"
    system_role_name: ClassVar[str]  = "system"


    messages: list[ChatMessageItem] = Field(default=[], description="List of chat messages in the conversation.")
    model: Optional[str] = Field(default=None, description="The model used for the chat conversation.")
    
    # option fields
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature for the model.")
    response_format: Optional[dict] = Field(default=None, description="Format of the response from the model.")
    
    def add_image_message_by_path(self, role: str, content:str, image_path: str) -> None:
        """
        Add an image message to the chat history using a local image file path.
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The text content of the message.
            image_path (str): The local file path to the image.
        """
        if not role or not image_path:
            logger.error("Role and image path must be provided.")
            return
        # Convert local image path to data URL
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        # Encode the image data to base64
        if isinstance(image_data, bytes):
            image_data = base64.b64encode(image_data).decode('utf-8')
        # Create the image URL in data URL format
        mime_type = "image/jpeg"  # Assuming JPEG, adjust as necessary
        image_url = f"data:{mime_type};base64,{image_data}"
        self.add_image_message(role, content, image_url)

    def add_image_message(self, role: str, content: str, image_url: str) -> None:
        """
        Add an image message to the chat history.
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The text content of the message.
            image_url (str): The URL of the image to be included in the message.
        """
        
        if not role or not image_url:
            logger.error("Role and image URL must be provided.")
            return
        content_item = [
            ChatContentItem(type="image_url", image_url={"url": image_url})
        ]
        if content:
            content_item.append(ChatContentItem(type="text", text=content))

        self.messages.append(ChatMessageItem(role=role, content=content_item))
        logger.debug(f"Image message added: {role}: {image_url}")

    def append_image_to_last_message_by_path(self, role:str, image_path: str) -> None:
        """
        Append an image to the last message in the chat history using a local image file path.
        
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            image_path (str): The local file path to the image.
        """
        if not image_path:
            logger.error("Image path must be provided.")
            return
        # Convert local image path to data URL
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        # Encode the image data to base64
        if isinstance(image_data, bytes):
            image_data = base64.b64encode(image_data).decode('utf-8')
        # Create the image URL in data URL format
        mime_type = "image/jpeg"  # Assuming JPEG, adjust as necessary
        image_url = f"data:{mime_type};base64,{image_data}"
        self.append_image_to_last_message(role, image_url)

    def append_image_to_last_message(self, role:str, image_url: str) -> None:
        """
        Append an image to the last message in the chat history if the role matches.
        
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            image_url (str): The URL of the image to append to the last message.
        """
        if not self.messages:
            content = ChatContentItem(type="image_url", image_url={"url": image_url})
            self.messages.append(ChatMessageItem(role=role, content=[content]))
            logger.debug("No messages to append to. Added new message.")
            return
        
        last_message = self.messages[-1]
        if last_message.role != role:
            content = ChatContentItem(type="image_url", image_url={"url": image_url})
            self.messages.append(ChatMessageItem(role=role, content=[content]))
            logger.debug(f"Added new message as last message role '{last_message.role}'")
            return
        
        # Check if the last content is a list and contains an image item
        if isinstance(last_message.content, list):
            last_message.content.append(ChatContentItem(type="image_url", image_url={"url": image_url}))
            logger.debug(f"Added new image item to last message: {image_url}")
        else:
            logger.error("Last message content is not in expected format (list). Cannot append image.")

    def append_text_to_last_message(self, role:str, additional_text: str) -> None:
        """
        Append additional text to the last message in the chat history if the role matches.
        
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            additional_text (str): The text to append to the last message.
        """
        if not self.messages:
            content = ChatContentItem(type="text", text=additional_text)
            self.messages.append(ChatMessageItem(role=role, content=[content]))
            logger.debug("No messages to append to. Added new message.")
            return
        last_message = self.messages[-1]

        if last_message.role != role:
            content = ChatContentItem(type="text", text=additional_text)
            self.messages.append(ChatMessageItem(role=role, content=[content]))
            logger.debug(f"Added new message as last message role '{last_message.role}'")
            return

        # Check if the last content is a list and contains a text item
        if isinstance(last_message.content, list):
            # If no text item found, add a new text item
            last_message.content.append(ChatContentItem(type="text", text=additional_text))
            logger.debug(f"Added new text item to last message: {additional_text}")
        else:
            logger.error("Last message content is not in expected format (list). Cannot append text.")

    def add_text_message(self, role: str, content: str) -> None:
        """
        Add a message to the chat history.
        
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The content of the message.
        """
        if not role or not content:
            logger.error("Role and content must be provided.")
            return
        content_item = ChatContentItem(type="text", text=content)

        self.messages.append(ChatMessageItem(role=role, content=[content_item]))
        logger.debug(f"Message added: {role}: {content}")

    def get_last_message(self) -> Optional[ChatMessageItem]:
        """
        Get the last message in the chat history.
        
        Returns:
            Optional[MessageItem]: The last message or None if no messages exist.
        """
        if self.messages:
            last_message = self.messages[-1]
            logger.debug(f"Last message retrieved: {last_message}")
            return last_message
        else:
            logger.debug("No messages found.")
            return None

    def add_messages(self, messages: list[ChatMessageItem]) -> None:
        """
        Add multiple messages to the chat history.
        
        Args:
            messages (list[dict]): A list of message dictionaries to add.
        """
        if not messages:
            logger.error("No messages provided to add.")
            return
        self.messages.extend(messages)
        logger.debug(f"Added {len(messages)} messages to chat history.")            

    def to_dict(self) -> dict:
        """
        Convert the chat messages to a dictionary format.
        
        Returns:
            dict: A dictionary representation of the chat messages.
        """
        params = {}
        params["messages"] = self.messages
        params["model"] = self.model
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.response_format is not None:
            params["response_format"] = self.response_format
        logger.debug(f"Converting chat messages to dict: {params}")
        return params

class CompletionResponse(BaseModel):
    output: str = Field(default="", description="The output text from the chat model.")
    total_tokens: int = Field(default=0, description="The total number of tokens used in the chat interaction.")

class OpenAIProps(BaseModel):

    openai_key: str = Field(default=os.getenv("OPENAI_API_KEY",""), alias="openai_key")
    azure_openai: bool = Field(default=os.getenv("AZURE_OPENAI","false").lower() == "true", alias="azure_openai")
    azure_openai_api_version: Optional[str] = Field(default=os.getenv("AZURE_OPENAI_API_VERSION",""), alias="azure_openai_api_version")
    azure_openai_endpoint: Optional[str] = Field(default=os.getenv("AZURE_OPENAI_ENDPOINT",""), alias="azure_openai_endpoint")
    openai_base_url: Optional[str] = Field(default=os.getenv("OPENAI_BASE_URL",""), alias="openai_base_url")

    completion_model: str = Field(default=os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4o"), alias="default_completion_model")
    embedding_model: str = Field(default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), alias="default_embedding_model")

    @model_validator(mode='before')
    def handle_azure_openai_bool_and_version(cls, values):
        azure_openai = values.get("azure_openai", False)
        if isinstance(azure_openai, str):
            values["azure_openai"] = azure_openai.upper() == "TRUE"
        if values.get("azure_openai_api_version") is None:
            values["azure_openai_api_version"] = "2024-02-01"
        return values

    def create_client_params(self) -> dict:
        if self.azure_openai:
            return self.__create_azure_openai_dict()
        else:
            return self.__create_openai_dict()
        
    def __create_openai_dict(self) -> dict:
        completion_dict = {}
        completion_dict["api_key"] = self.openai_key
        if self.openai_base_url:
            completion_dict["base_url"] = self.openai_base_url
        return completion_dict

    def __create_azure_openai_dict(self) -> dict:
        completion_dict = {}
        completion_dict["api_key"] = self.openai_key
        if self.openai_base_url:
            completion_dict["base_url"] = self.openai_base_url
        else:
            completion_dict["azure_endpoint"] = self.azure_openai_endpoint
            completion_dict["api_version"] = self.azure_openai_api_version
        return completion_dict

    @staticmethod
    def check_env_vars() -> bool:
        # OPENAI_API_KEYの存在を確認
        if "OPENAI_API_KEY" not in os.environ:
            logger.error("OPENAI_API_KEY is not set in the environment variables.")
            return False
        # AZURE_OPENAIの存在を確認
        if "AZURE_OPENAI" not in os.environ:
            logger.error("AZURE_OPENAI is not set in the environment variables.")
            return False
        if os.environ.get("AZURE_OPENAI", "false").lower() == "true":
            # AZURE_OPENAI_API_VERSIONの存在を確認
            if "AZURE_OPENAI_API_VERSION" not in os.environ:
                logger.error("AZURE_OPENAI_API_VERSION is not set in the environment variables.")
                return False
            # AZURE_OPENAI_ENDPOINTの存在を確認
            if "AZURE_OPENAI_ENDPOINT" not in os.environ:
                logger.error("AZURE_OPENAI_ENDPOINT is not set in the environment variables.")
                return False
        
        # DEFAULT_COMPLETION_MODELの存在を確認
        if "OPENAI_COMPLETION_MODEL" not in os.environ:
            logger.warning("OPENAI_COMPLETION_MODEL is not set in the environment variables. Defaulting to 'gpt-4o'.")
        # DEFAULT_EMBEDDING_MODELの存在を確認
        if "OPENAI_EMBEDDING_MODEL" not in os.environ:
            logger.warning("OPENAI_EMBEDDING_MODEL is not set in the environment variables. Defaulting to 'text-embedding-3-small'.")
        return True
    
    @staticmethod
    def local_image_to_data_url(image_path) -> str:
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"

    @staticmethod
    def create_openai_chat_parameter_dict_simple(model: str, prompt: str, temperature: Union[float, None] = 0.5, json_mode: bool = False) -> dict:
        messages = [{"role": "user", "content": prompt}]
        params: dict[str, Any] = {}
        params["messages"] = messages
        params["model"] = model
        if temperature:
            params["temperature"] = temperature
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        return params


import json
from openai import AsyncOpenAI, AsyncAzureOpenAI
from pydantic import BaseModel, Field
from typing import Optional, Any

class LLMClient:
    def __init__(self, props: OpenAIProps):
        
        self.props = props

    def __get_completion_client(self) -> Union[AsyncOpenAI, AsyncAzureOpenAI]:
        
        if (self.props.azure_openai):
            params = self.props.create_client_params()
            return AsyncAzureOpenAI(
                **params
            )

        else:
            params =self.props.create_client_params()
            return AsyncOpenAI(
                **params
            )

    async def run_completion_async(self, input_dict: CompletionRequest) -> CompletionResponse:
        # openai.
        # RateLimitErrorが発生した場合はリトライする
        # リトライ回数は最大で3回
        # リトライ間隔はcount*30秒
        # リトライ回数が5回を超えた場合はRateLimitErrorをraiseする
        # リトライ回数が5回以内で成功した場合は結果を返す
        # OpenAIのchatを実行する
        completion_client = self.__get_completion_client()
        count = 0
        response = None
        while count < 3:
            try:
                response = await completion_client.chat.completions.create(
                    **input_dict.to_dict()
                )
                break
            except RateLimitError as e:
                count += 1
                # rate limit errorが発生した場合はリトライする旨を表示。英語
                logger.warn(f"RateLimitError has occurred. Retry after {count*30} seconds.")
                time.sleep(count*30)
                if count == 5:
                    raise e
        if response is None:
            raise RuntimeError("Failed to get a response from OpenAI after retries.")
        # token情報を取得する
        total_tokens = response.usage.total_tokens
        # contentを取得する
        content = response.choices[0].message.content

        # dictにして返す
        logger.info(f"chat output:{json.dumps(content, ensure_ascii=False, indent=2)}")
        return CompletionResponse(output=content, total_tokens=total_tokens)
