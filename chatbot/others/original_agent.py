import json
import os
from collections.abc import Generator
from typing import Any, Optional

import tiktoken
from common.openai_tools.basetool import BaseTool
from openai import AzureOpenAI
from qdrant_client import QdrantClient

class Agent:
    def __init__(
        self,
        tools: list[BaseTool],
        memory: list,
        base_prompt: str,
        tool_name_dict: Optional[dict[str, BaseTool]] = None,
        model: str = "gpt4",
        function_call: str = "auto",
        temperature: float = 0.0,
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

    def _remove_old_memory(self):
        memory_token_limit = 6500 if self.model == "gpt4-turbo" else 14000

        all_memory_in_string = " ".join([m["content"] for m in self.memory])
        all_memory_in_tokens = len(self._encoder.encode(all_memory_in_string))

        while all_memory_in_tokens > memory_token_limit:
            del self.memory[1:3]
            all_memory_in_string = " ".join([m["content"] for m in self.memory])
            all_memory_in_tokens = len(self._encoder.encode(all_memory_in_string))

    def _response_with_tools(self):
        client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("OPENAI_API_BASE"),
        )

        has_tools = len(self.tools) > 0
        try:
            if has_tools:
                yield from client.chat.completions.create(
                    model=self.model,
                    messages=self.memory,
                    functions=self.openai_schemas,
                    function_call=self.function_call,
                    temperature=self.temperature,
                    stream=True,
                )
            else:
                yield from client.chat.completions.create(
                    model=self.model,
                    messages=self.memory,
                    temperature=self.temperature,
                    stream=True,
                )
        except Exception as e:
            if "ResponsibleAIPolicyViolation" in str(e):
                yield from " Your question seems to be in violation of the responsible AI policy. Please try and reformulate your question."
            else:
                yield from " Sorry, I am unable to answer your question. Please try and reformulate your question."

    def get_delta_and_finish_reason(self, response: object) -> object:
        has_choices = filter(lambda chunk: len(chunk.choices), response)
        first_choice = (chunk.choices[0] for chunk in has_choices)
        delta_and_finish_reason = ((chunk.delta, chunk.finish_reason) for chunk in first_choice)

        return delta_and_finish_reason

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

        delta_and_finish_reason = self.get_delta_and_finish_reason(response)

        response_content = ""

        for delta, _ in delta_and_finish_reason:
            if delta.function_call:
                break

            if delta.content is not None:
                if self.formulated_final_answer:
                    yield delta.content
                response_content += delta.content
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

        funcname_to_call = delta.function_call.name
        tool_to_call: BaseTool = self.tools_dict[funcname_to_call]

        argument_chunks = filter(lambda tpl: tpl[1] is None, delta_and_finish_reason)
        arg_strings = (tpl[0].function_call.arguments for tpl in argument_chunks)
        arguments_as_string = "".join(arg_strings)
        response_function_args_dict = json.loads(arguments_as_string)

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