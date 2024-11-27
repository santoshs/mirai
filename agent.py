"""This module defines an `Agent` class.

Create agents to perform task delegation and manage tool calls.

Key Components:
- Agent class: Initializes an agent with a specific role, role description,
  and a set of functions (tools) it can utilize.
- Function Metadata Extraction: The Agent dynamically extracts metadata from
  each function to structure prompts.
- Tool Execution: The Agent can execute tool calls and recursively handle
  follow-up calls if needed.

Example usage:
    agent = Agent(role="Support agent",
                  role_description="Help customer", ...)
    response = agent("Hello")
"""

import json
import logging
from openai import OpenAI
from pydantic import BaseModel, Field, PrivateAttr
from typing import Callable, List, Dict, Any, Optional

from utils import get_function_info


class Agent(BaseModel):
    """Create an AI agent.

    A class representing an agent that can take up specific roles and also
    manage tool calls.
    """

    role: str = Field(...)
    role_description: str = Field(...)
    functions: List[Callable] = Field(default_factory=list)
    api_key: str = Field(...)
    base_url: Optional[str] = Field(default=None)
    model: str = Field(...)
    messages: List[Dict[str, str]] = Field(default_factory=list)

    _available_functions: Dict[str, Callable] = PrivateAttr()
    _functions_info: List[Dict[str, Any]] = PrivateAttr()
    _client: OpenAI = PrivateAttr()

    def __init__(self, **data):
        """Initialize the Agent with the provided data."""
        super().__init__(**data)
        self._available_functions = {
            func.__name__: func for func in self.functions
        }
        self._functions_info = [
            get_function_info(func) for func in self.functions
        ]

        self.messages = [{
            'role': 'system',
            'content': f"You are a {self.role}. {self.role_description}"
        }]
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def call_llm(self, messages: List[Dict[str, str]]) -> Any:
        """Get the models response based on the provided messages.

        Args:
            messages (List[Dict[str, str]]): The messages to pass to the API.

        Returns:
            Any: The API response or an error message.
        """
        try:
            if len(self.functions) > 0:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self._functions_info,
                    tool_choice="auto",
                )
            else:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
        except Exception as e:
            return f"Error during API call: {e}"

    def execute_tool_call(self, tool_call) -> tuple:
        """Execute a single tool call and return result or error.

        Args:
            tool_call: The tool call data containing the function and arguments.

        Returns:
            tuple: A tuple containing the tool ID and the result.
        """
        logging.debug("executing tool call")
        tool_id = tool_call.id
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if function_name in self._available_functions:
            func = self._available_functions[function_name]
            try:
                # Call the function with provided arguments
                result = func(**arguments)
                return tool_id, result
            except Exception as e:
                return tool_id, \
                    f"Error executing function '{function_name}': {e}"
        return tool_id, f"Function '{function_name}' is not available."

    def process_tool_calls(self, tool_calls) -> Dict[str, Any]:
        """Process each tool call.

        Execute the corresponding function, and store results.

        Args:
            tool_calls: A list of tool calls to process.

        Returns:
            Dict[str, Any]: A dictionary with results of each tool call.
        """
        outputs = {}
        for tool_call in tool_calls:
            tool_id, result = self.execute_tool_call(tool_call)
            outputs[tool_id] = result
            # Append each tool call result to messages with the role 'function'
            self.messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": result
            })

        return outputs

    def follow_up_with_results(self) -> str:
        """Send follow-up request to model with tool results.

        Returns:
            str: The assistant's final response.
        """
        logging.debug("Following up with response")
        follow_up_response = self.call_llm(self.messages)
        if isinstance(follow_up_response, str):
            logging.debug(follow_up_response)
            return follow_up_response

        final_response_message = follow_up_response.choices[0].message
        logging.debug(final_response_message)
        tool_calls = final_response_message.tool_calls

        if tool_calls and len(tool_calls) != 0:
            # Process all tool calls and log the results
            self.messages.append(final_response_message)
            self.process_tool_calls(tool_calls)
            return self.follow_up_with_results()

        # No function calls, just return the assistant's response
        assistant_response = final_response_message.content.strip()
        self.messages.append({
            "role": "assistant",
            "content": assistant_response
        })
        return assistant_response

    def __call__(self, user_input: str) -> str:
        """Generate response to the user's input using modularized functions.

        Args:
            user_input (str): The input message from the user.

        Returns:
            str: The response generated by the assistant.
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_input})

        # Call LLM API for the initial response
        response = self.call_llm(self.messages)
        if isinstance(response, str):
            # Return error if initial API call failed
            error_message = response
            self.messages.append({
                "role": "assistant",
                "content": error_message
            })
            return error_message

        logging.debug(response)
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls and len(tool_calls) != 0:
            # Process all tool calls and log the results
            self.messages.append(response_message)
            self.process_tool_calls(tool_calls)
            return self.follow_up_with_results()

        # No function calls, just return the assistant's response
        assistant_response = response_message.content.strip()
        self.messages.append({
            "role": "assistant",
            "content": assistant_response
        })
        return assistant_response

    class Config:
        """Config for pydantic."""

        arbitrary_types_allowed = True
