"""Base class for agent and manger."""

from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing import Callable, List, Dict, Any, Optional
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace import Tracer
from opentelemetry import trace
from openai import OpenAI
import json
import sys

from .utils import get_function_info

# Check Python version and define Self accordingly
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing import TypeVar
    T = TypeVar("T", bound="BaseAgent")
    Self = T


class BaseAgent(BaseModel):
    """Base class for agent and Manager."""

    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    model: Optional[str] = Field(default=None)
    messages: List[Dict[str, str]] = Field(default_factory=list)
    functions: List[Callable] = Field(default_factory=list)
    run: Optional[Callable] = Field(default=None)

    _available_functions: Dict[str, Callable] = PrivateAttr()
    _functions_info: List[Dict[str, Any]] = PrivateAttr()
    _tracer: Tracer = PrivateAttr(default_factory=None)
    _client: OpenAI = PrivateAttr()
    _complete: bool = PrivateAttr(default=False)
    _iterative_mode: bool = PrivateAttr(default=False)
    _system_prompt: str = PrivateAttr(default="You are a helpful assistant.")
    _metadata: Dict = PrivateAttr(default={})

    @model_validator(mode='after')
    def check_fields(self) -> Self:
        """Ensure proper values are provided."""
        if not self.run and not (self.api_key and self.model):
            raise ValueError("Either 'run' must be provided, or both 'api_key' and 'model' must be specified.")

        return self

    def __init__(self, **data):
        """Initialize the Agent with the provided data."""
        super().__init__(**data)

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._tracer = trace.get_tracer(__name__)

    def _call_llm(self) -> Any:
        """Get the models response based on the provided messages."""
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
            current_span = trace.get_current_span()
            current_span.set_status(Status(StatusCode.ERROR))
            return f"Error during API call: {e}"

    def _execute_tool_call(self, tool_call) -> tuple:
        """Execute a single tool call and return result or error."""
        tool_id = tool_call.id
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if "tool_calls" not in self.metadata.keys():
            self.metadata["tool_calls"] = []

        span = trace.get_current_span()
        span.set_attribute(SpanAttributes.TOOL_NAME,
                           function_name)
        span.set_attribute(SpanAttributes.TOOL_PARAMETERS,
                           tool_call.function.arguments)

        if function_name in self._available_functions:
            func = self._available_functions[function_name]
            with self._tracer.start_as_current_span(
                    f"Calling {function_name}") as span:
                try:
                    # Call the function with provided arguments
                    result = func(**arguments)
                    span.set_attribute(SpanAttributes.OUTPUT_VALUE, result)
                    span.set_status(Status(StatusCode.OK))

                    # Save result to metadata
                    self.metadata["tool_calls"].append({
                        "id": tool_id,
                        "function": function_name,
                        "arguments": arguments,
                        "output": result,
                        "status": "success"
                    })

                    return tool_id, result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR))
                    self.metadata["tool_calls"].append({
                        "id": tool_id,
                        "function": function_name,
                        "arguments": arguments,
                        "output": str(e),
                        "status": "error"
                    })

                    return tool_id, \
                        f"Error executing function '{function_name}': {e}"

        error_message = f"Function '{function_name}' is not available."
        self.metadata["tool_calls"].append({
            "id": tool_id,
            "function": function_name,
            "arguments": arguments,
            "output": error_message,
            "status": "error"
        })
        return tool_id, f"Function '{function_name}' is not available."

    def _process_tool_calls(self, tool_calls) -> Dict[str, Any]:
        """Process each tool call."""
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
            return None

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        assistant_response = response_message.content

        if tool_calls and len(tool_calls) != 0:
            with self._tracer.start_as_current_span(
                    "Processing tool calls") as span:
                # Process tool calls
                self.messages.append(response_message)
                self._process_tool_calls(tool_calls)
                span.set_status(Status(StatusCode.OK))
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
            span.set_status(Status(StatusCode.OK))
            return result

    def __call__(self, user_input: str) -> str:
        """Generate response to the user's input."""
        if self.run:
            return self.run(user_input)

        # Add user message
        with self._tracer.start_as_current_span("processing_query") as span:
            span.set_attribute(SpanAttributes.INPUT_VALUE, user_input)
            self.messages.append({"role": "user", "content": user_input})

            while True:
                # Initial response from LLM
                response = self._call_llm()

                # Process response and decide on follow-up
                result = self._process_response(response)
                if result is not None:
                    span.set_attribute(SpanAttributes.OUTPUT_VALUE,
                                       response.choices[0].message.content)
                    span.set_status(Status(StatusCode.OK))
                    return result  # Final response or error
                if not self._iterative_mode:
                    return self._follow_up_with_results()

    def clear_history(self):
        """Clear history."""
        self.messages = [{
            'role': 'system',
            'content': self._system_prompt
        }]

        return

    def add_history(self, messages):
        """Add history."""
        self.messages.extend(messages)

    @property
    def metadata(self) -> Dict:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict):
        if not isinstance(value, dict):
            raise ValueError("metadata must be a dictionary.")

        self._metadata = value
