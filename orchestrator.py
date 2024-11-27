from typing import List, Dict, Optional, Callable, Any
from pydantic import BaseModel, Field, PrivateAttr
from openai import OpenAI
import logging
import json

from agent import Agent
from opentelemetry import trace
from utils import get_function_info

ORCHESTRATOR_PROMPT = """You are the Orchestrator in a multi-agent system,
responsible for analysing, coordinating, and delegating tasks among various
specialized agents. Each agent has a unique role and expertise, and your job is
to interpret task requests, select the best-suited agent(s) for each task, and
ensure seamless communication between agents when collaboration is
required. When the response is complete, call the complete tool to indicate
task completion. Do not offer any additional assistance or ask if further
clarification is needed; simply call complete if done."

Responsibilities:

- Analyze and Route Tasks: Carefully read the task descriptions provided and determine which agent or group of agents should handle each request.
- Delegate Responsibilities: Based on the nature of the task, either delegate it to a single agent, broadcast it to multiple agents for feedback, or coordinate complex tasks that require input from multiple agents.
- Facilitate Agent Communication: When necessary, relay information between agents to ensure tasks are handled efficiently, and relevant insights or updates are shared effectively.
- Provide Context: Offer each agent relevant task context, helping them understand the problem at hand so they can respond appropriately.
- Summarize Results: Gather responses from agents and present consolidated, clear answers or next steps.

Agents:
{}

Goal:

Your goal is to effectively manage resources, improve response times, and
ensure high-quality, coordinated answers by routing tasks to the right agents
and facilitating smooth communication when multiple agents are involved by
using the tools provided.
"""


class AgentManager(BaseModel):
    """Manage and coordinate a group of agents.

    LLM orchestrator to facilitate task delegation and inter-agent
    communication.
    """

    api_key: str = Field(...)
    base_url: Optional[str] = Field(default=None)
    model: str = Field(...)
    agents: List[Callable] = Field(default_factory=list)
    messages: List[Dict[str, str]] = Field(default_factory=list)

    _functions: List[Callable] = PrivateAttr(default_factory=list)
    _available_functions: Dict[str, Callable] = PrivateAttr()
    _functions_info: List[Dict[str, Any]] = PrivateAttr()
    _client: OpenAI = PrivateAttr()
    agent_messages: str = Field(default="")
    _tracer: trace.Tracer = PrivateAttr(default_factory=None)

    def __init__(self, **data):
        """Initialize the Agent with the provided data."""
        super().__init__(**data)
        self._functions = [self.transfer_to_agent, self.complete]
        self._available_functions = {
            func.__name__: func for func in self._functions
        }
        self._functions_info = [
            get_function_info(func) for func in self._functions
        ]

        self._agents = {agent.role.lower(): agent for agent in self.agents}
        self._complete = False

        agents = ""
        for agent in self.agents:
            agents += f"\n- {agent.role}"

        self.messages = [{
            'role': 'system',
            'content': ORCHESTRATOR_PROMPT.format(agents)
          }]
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.agent_messages = ""
        self._tracer = trace.get_tracer(__name__)
        self._tracer.start_as_current_span("Processing Query")

        print(self._functions_info)

    def call_llm(self, messages: List[Dict[str, str]]) -> Any:
        """Get the models response based on the provided messages.

        Args:
            messages (List[Dict[str, str]]): The messages to pass to the API.

        Returns:
            Any: The API response or an error message.
        """
        try:
            return self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self._functions_info,
                tool_choice="auto",
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
            self._tracer.start_as_current_span(f"Trying to call agent {func.__name__}")
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
            content = self.process_tool_calls(tool_calls)
            print(list(content.values())[-1])
            if self._complete:
                return list(content.values())[-1]

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

        while True:
            # Call LLM for the initial response
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
                content = self.process_tool_calls(tool_calls)
                print(list(content.values())[-1])
                if self._complete:
                    return list(content.values())[-1]

                self.follow_up_with_results()
                continue

            # No function calls, just return the assistant's response
            assistant_response = response_message.content.strip()
            self.messages.append({
                "role": "assistant",
                "content": assistant_response
            })

    def transfer_to_agent(self, role: str, task: str) -> Optional[str]:
        """Delegate a task to an agent based on its role."""
        agent = self._agents.get(role.lower())
        if not agent:
            return f"No agent found with role '{role}'."

        agent_response = agent(task)
        self.agent_messages += agent_response

        return agent_response

    def complete(self, text: str):
        """Signal that the work is complete."""
        self._complete = True
        print(self.agent_messages)
        return text

    class Config:
        """Config for pydantic."""

        arbitrary_types_allowed = True
