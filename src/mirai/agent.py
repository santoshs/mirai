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

from pydantic import Field

from .base import BaseAgent


class Agent(BaseAgent):
    """Create an AI agent.

    A class representing an agent that can take up specific roles and also
    manage tool calls.
    """

    role: str = Field(...)
    role_description: str = Field(...)

    def __init__(self, **data):
        """Initialize the Agent with the provided data."""
        super().__init__(**data)
        self.messages = [{
            'role': 'system',
            'content': f"You are a {self.role}. {self.role_description}"
        }]
