import json
import os
from collections.abc import Generator
from typing import Any, Optional

import tiktoken
from common.openai_tools.basetool import BaseTool
from qdrant_client import QdrantClient

# Add imports for IBM Watson integration
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
import litellm
from litellm.types.utils import ModelResponse, Message
from litellm import completion
from instructor import from_litellm, Mode
from pydantic import BaseModel, create_model, Field
from decouple import config

# Define base response model for LLMCaller
class BaseResponse(BaseModel):
    """A default response model that defines a single
    field `answer` to store the response from the LLM."""
    answer: str

# LLMCaller class for IBM Watson integration
class LLMCaller:
    """A class to interact with an LLM using the LiteLLM and Instructor
    libraries. This class is designed to simplify the process of sending
    prompts to an LLM and receiving structured responses."""

    def __init__(
        self,
        api_key: str,
        project_id: str,
        api_url: str,
        model_id: str = "watsonx/ibm/granite-3-8b-instruct",
        params: dict[str, Any] = None,
    ):
        """
        Initializes the LLMCaller instance with the necessary credentials and configuration.

        Args:
            api_key (str): The API key for authenticating with the LLM service.
            project_id (str): The project ID associated with the LLM service.
            api_url (str): The base URL for the LLM service API.
            model_id (str): The identifier of the specific LLM model to use.
            params (dict[str, Any]): Additional parameters to configure the LLM's behavior.
        """
        self.api_key = api_key
        self.project_id = project_id
        self.api_url = api_url
        self.model_id = model_id
        self.params = params or {
            GenParams.TEMPERATURE: 0.6,
            GenParams.MAX_NEW_TOKENS: 300,
        }

        # Boilerplate: Configure LiteLLM to drop unsupported parameters for Watsonx.ai
        litellm.drop_params = True
        # Boilerplate: Create an Instructor client for pydantic-based interactions with the LLM
        self.client = from_litellm(completion, mode=Mode.JSON)

    def create_response_model(self, title: str, fields: dict):
        """Dynamically creates a Pydantic response model for the LLM's output."""
        return create_model(title, **fields, __base__=BaseResponse)

    def invoke(self, prompt: str, response_model=BaseResponse, **kwargs):
        """Sends a prompt to the LLM and retrieves a structured response."""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                    + "\n\n"
                    + f"Provide your answer as an object of {type(response_model)}",
                }
            ],
            project_id=self.project_id,
            apikey=self.api_key,
            api_base=self.api_url,
            response_model=response_model,
            **kwargs,
            **self.params,
        )
        return response

    def chat(self, messages: list[dict[str, str] | Message], **kwargs):
        """Sends a prompt to the LLM without a structured response."""
        return completion(
            model=self.model_id,
            project_id=self.project_id,
            apikey=self.api_key,
            api_base=self.api_url,
            messages=messages,
            **kwargs,
            **self.params,
        )

class Agent:
    def __init__(
        self,
        tools: list[BaseTool],
        memory: list,
        base_prompt: str,
        tool_name_dict: Optional[dict[str, BaseTool]] = None,
        model: str = "watsonx/ibm/granite-3-8b-instruct",
        function_call: str = "auto",
        temperature: float = 0.6,
        tool_response_token_size: int = 3500,
        max_calls: int = 10,
        improve_final_answer: bool = False,
    ):
        self.tools = tools
        self.tool_name_dict = tool_name_dict
        self.memory = memory
        self.base_prompt = base_prompt
        self.model = model
        self.function_call = function_call
        self.temperature = temperature
        self.tool_response_token_size = tool_response_token_size
        self.max_calls = max_calls
        self.initial_query = None
        self.improve_final_answer = improve_final_answer
        self.formulated_final_answer = False

        if len(self.memory) == 0:
            self.memory.insert(0, {"role": "system", "content": self.base_prompt})

        if len(self.memory) >= 1 and self.memory[0]["role"] != "system":
            self.memory.insert(0, {"role": "system", "content": self.base_prompt})

        self._encoder = tiktoken.get_encoding("cl100k_base")
        self.tools_dict: dict[str, BaseTool] = {tool.__name__: tool for tool in self.tools}
        self.openai_schemas = [tool.openai_schema for tool in self.tools]
        self._tool_outputs = {}
        
        # Initialize LLMCaller
        self.llm_caller = LLMCaller(
            api_key=os.getenv("WX_API_KEY"),
            project_id=config("WX_PROJECT_ID"),
            api_url=os.getenv("WX_API_URL", "https://us-south.ml.cloud.ibm.com"),
            model_id=self.model,
            params={
                GenParams.TEMPERATURE: self.temperature,
                GenParams.MAX_NEW_TOKENS: 300,
            }
        )

    def _remove_old_memory(self):
        memory_token_limit = 6500 if "granite-3-8b" in self.model else 14000

        all_memory_in_string = " ".join([m["content"] for m in self.memory])
        all_memory_in_tokens = len(self._encoder.encode(all_memory_in_string))

        while all_memory_in_tokens > memory_token_limit:
            del self.memory[1:3]
            all_memory_in_string = " ".join([m["content"] for m in self.memory])
            all_memory_in_tokens = len(self._encoder.encode(all_memory_in_string))

    def _response_with_tools(self):
        has_tools = len(self.tools) > 0
        try:
            if has_tools:
                # Using LLMCaller with tools
                response = self.llm_caller.chat(
                    messages=self.memory,
                    functions=self.openai_schemas,
                    function_call=self.function_call,
                )
                yield response
            else:
                # Using LLMCaller without tools
                response = self.llm_caller.chat(
                    messages=self.memory,
                )
                yield response
        except Exception as e:
            if "ResponsibleAIPolicyViolation" in str(e):
                yield " Your question seems to be in violation of the responsible AI policy. Please try and reformulate your question."
            else:
                yield " Sorry, I am unable to answer your question. Please try and reformulate your question."

    def get_delta_and_finish_reason(self, response: object) -> object:
        # Adapt for LLMCaller response format
        if isinstance(response, ModelResponse):
            # Extract the relevant parts from the ModelResponse
            choices = response.choices if hasattr(response, 'choices') else []
            if choices:
                delta = choices[0].delta if hasattr(choices[0], 'delta') else None
                finish_reason = choices[0].finish_reason if hasattr(choices[0], 'finish_reason') else None
                return [(delta, finish_reason)]
            return []
        
        # If it's a string (error message)
        if isinstance(response, str):
            return [({"content": response}, None)]
        
        # Default case
        return [({"content": "I'm unable to process your request."}, None)]

    def generate_response(
        self,
        question: str,
        role: str = "user",
        response_function: str | None = None,
        num_calls: int = 0,
        qdrant_client: QdrantClient = None,
    ) -> Generator[str, None, None]:
        if num_calls > self.max_calls:
            return "Sorry, I am unable to answer your question. Please try and reformulate your question."

        if num_calls == 0:
            self.initial_query = question

        memory = {"role": role, "content": question}
        if role == "function":
            memory["name"] = response_function

        self.memory.append(memory)

        self._remove_old_memory()

        response = self._response_with_tools()

        # Get the first response object
        try:
            response_obj = next(response)
            delta_and_finish_reason = self.get_delta_and_finish_reason(response_obj)
        except StopIteration:
            delta_and_finish_reason = [({"content": "I'm unable to process your request."}, None)]

        response_content = ""

        for delta, finish_reason in delta_and_finish_reason:
            if delta.get("function_call"):
                break

            if delta.get("content") is not None:
                content = delta.get("content", "")
                if self.formulated_final_answer:
                    yield content
                response_content += content
            else:
                if self.improve_final_answer and not self.formulated_final_answer:
                    self.formulated_final_answer = True
                    yield from self.generate_response(
                        role="assistant",
                        question=f"Given the initial question: {self.initial_query}, and the information you have, please generate a response that answers the initial question to the best of your ability, while highlighting the data used to answer. Also make sure no information is made up.",
                        num_calls=num_calls + 1,
                        qdrant_client=qdrant_client,
                    )
                else:
                    if self.formulated_final_answer:
                        del self.memory[-1]
                    self.memory.append({"role": "assistant", "content": response_content})
                return

        # Handle function calls
        function_call = delta.get("function_call", {})
        funcname_to_call = function_call.get("name")
        tool_to_call: BaseTool = self.tools_dict[funcname_to_call]

        # Extract arguments
        arguments_as_string = function_call.get("arguments", "{}")
        try:
            response_function_args_dict = json.loads(arguments_as_string)
        except json.JSONDecodeError:
            response_function_args_dict = {}

        if self.tool_name_dict:
            yield f"**Using {self.tool_name_dict[funcname_to_call]} data to generate answer...**\n\n"
        else:
            yield "**Looking for answer in my internal knowledge...**\n\n"

        tool_response, tool_data = tool_to_call(**response_function_args_dict).run(qdrant_client)

        self._tool_outputs[self.tool_name_dict[funcname_to_call]] = tool_data

        self.memory.append(
            {
                "role": "assistant",
                "function_call": {
                    "name": funcname_to_call,
                    "arguments": arguments_as_string,
                },
                "content": "",
            },
        )
        yield from self.generate_response(
            role="function",
            response_function=funcname_to_call,
            question=tool_response,
            num_calls=num_calls + 1,
            qdrant_client=qdrant_client,
        )

    @property
    def latest_tool_outputs(self) -> list[dict[str, Any]]:
        return self._tool_outputs