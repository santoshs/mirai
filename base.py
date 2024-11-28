"""Base class for agent and manger."""

from pydantic import BaseModel, Field, PrivateAttr
from typing import Callable, List, Dict, Any, Optional
from opentelemetry.trace import Tracer
from opentelemetry import trace
from openai import OpenAI
import json

from utils import get_function_info


class BaseAgent(BaseModel):
    """Base class for agent and Manager."""

    api_key: str = Field(...)
    base_url: Optional[str] = Field(default=None)
    model: str = Field(...)
    messages: List[Dict[str, str]] = Field(default_factory=list)
    functions: List[Callable] = Field(default_factory=list)
    run: Optional[Callable] = Field(default=None)

    _available_functions: Dict[str, Callable] = PrivateAttr()
    _functions_info: List[Dict[str, Any]] = PrivateAttr()
    _tracer: Tracer = PrivateAttr(default_factory=None)
    _client: OpenAI = PrivateAttr()
    _complete: bool = PrivateAttr(default=False)
    _iterative_mode: bool = PrivateAttr(default=False)

    def __init__(self, **data):
        """Initialize the Agent with the provided data."""
        super().__init__(**data)

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._tracer = trace.get_tracer(__name__)

    def _call_llm(self) -> Any:
        """Get the models response based on the provided messages.

        Args:
            messages (List[Dict[str, str]]): The messages to pass to the API.

        Returns:
            Any: The API response or an error message.
        """
        self._functions_info = [
            get_function_info(func) for func in self.functions
        ]
        self._available_functions = {
            func.__name__: func for func in self.functions
        }

        try:
            if len(self.functions) > 0:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self._functions_info,
                    tool_choice="auto",
                )
            else:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                )
        except Exception as e:
            return f"Error during API call: {e}"

    def _execute_tool_call(self, tool_call) -> tuple:
        """Execute a single tool call and return result or error.

        Args:
            tool_call: The tool call data containing the function and arguments.

        Returns:
            tuple: A tuple containing the tool ID and the result.
        """
        tool_id = tool_call.id
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if function_name in self._available_functions:
            func = self._available_functions[function_name]
            with self._tracer.start_as_current_span(
                    f"Calling {func.__name__}") as span:
                try:
                    # Call the function with provided arguments
                    result = func(**arguments)
                    span.set_attribute("agent.response", result)
                    return tool_id, result
                except Exception as e:
                    return tool_id, \
                        f"Error executing function '{function_name}': {e}"

        return tool_id, f"Function '{function_name}' is not available."

    def _process_tool_calls(self, tool_calls) -> Dict[str, Any]:
        """Process each tool call.

        Execute the corresponding function, and store results.

        Args:
            tool_calls: A list of tool calls to process.

        Returns:
            Dict[str, Any]: A dictionary with results of each tool call.
        """
        outputs = {}
        for tool_call in tool_calls:
            tool_id, result = self._execute_tool_call(tool_call)
            outputs[tool_id] = result
            # Append each tool call result to messages with the role 'function'
            self.messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": result
            })

        return outputs

    def _process_response(self, response) -> str:
        """Process LLM response and handle tool calls."""
        if isinstance(response, str):
            # Handle error response
            self.messages.append({"role": "assistant", "content": response})
            return response

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        assistant_response = response_message.content

        if tool_calls and len(tool_calls) != 0:
            with self._tracer.start_as_current_span("Processing tool calls"):
                # Process tool calls
                self.messages.append(response_message)
                self._process_tool_calls(tool_calls)
            return None  # Indicates further follow-up is required

        # Final response without tool calls
        self.messages.append({"role": "assistant",
                              "content": assistant_response.strip()})
        return assistant_response.strip()

    def _follow_up_with_results(self) -> str:
        """Recursive method to handle follow-up requests with tool results."""
        with self._tracer.start_as_current_span("response_followup") as span:
            response = self._call_llm()
            result = self._process_response(response)
            if result is None:  # If there are tool calls, continue recursion
                return self._follow_up_with_results()
            span.set_attribute("query.followup.response", result)
            return result

    def __call__(self, user_input: str) -> str:
        """Generate response to the user's input using modularized functions.

        Args:
            user_input (str): The input message from the user.

        Returns:
            str: The response generated by the assistant.
        """
        if self.run:
            return self.run(user_input)

        # Add user message
        with self._tracer.start_as_current_span("processing_query") as span:
            span.set_attribute("query.input", user_input)
            self.messages.append({"role": "user", "content": user_input})

            while True:
                # Initial response from LLM
                response = self._call_llm()

                # Process response and decide on follow-up
                result = self._process_response(response)
                if result is not None:
                    span.set_attribute("query.response",
                                       response.choices[0].message.content)
                    return result  # Final response or error
                if not self._iterative_mode:
                    return self._follow_up_with_results()
