"""The orchestrator agent."""

from typing import List, Callable
from pydantic import Field
from opentelemetry.trace import Status, StatusCode

from .base import BaseAgent


ORCHESTRATOR_PROMPT = """You are the Orchestrator in a multi-agent system,
responsible for analysing, coordinating, and delegating tasks among various
specialized agents. Each agent has a unique role and expertise, and your job is
to interpret task requests, select the best-suited agent(s) for each task, and
ensure seamless communication between agents when collaboration is required. Do
not offer any additional assistance or ask if further clarification is needed."

Responsibilities:

- Query Relevance: If there is a appropriate agent available, use it to check the relevance/validity of the query provided by the user.
- Analyze and Route Tasks: Carefully read the task descriptions provided and determine which agent or group of agents should handle each request.
- Delegate Responsibilities: Based on the nature of the task, either delegate it to a single agent, broadcast it to multiple agents for feedback, or coordinate complex tasks that require input from multiple agents.
- Facilitate Agent Communication: When necessary, relay information between agents to ensure tasks are handled efficiently, and relevant insights or updates are shared effectively.
- Provide Context: Offer each agent relevant task context, helping them understand the problem at hand so they can respond appropriately.
- Review Result: Review the provided solution/answer, if available by using one of the relevant agents.
- Summarize Results: Gather responses from agents and present consolidated, clear answers or next steps.

Agents:
{}

Goal:

Your goal is to effectively manage resources, improve response times, and
ensure high-quality, coordinated answers by routing tasks to the right agents
and facilitating smooth communication when multiple agents are involved by
using the tools provided.
"""


class AgentManager(BaseAgent):
    """Manage and coordinate a group of agents.

    LLM orchestrator to facilitate task delegation and inter-agent
    communication.
    """

    agents: List[Callable] = Field(default_factory=list)
    agent_messages: str = Field(default="")

    def __init__(self, **data):
        """Initialize the Agent with the provided data."""
        super().__init__(**data)
        self.functions = [self.transfer_to_agent]
        self._agents = {agent.role.lower(): agent for agent in self.agents}

        agents = ""
        for agent in self.agents:
            agents += f"\n- {agent.role}"

        self._system_prompt = ORCHESTRATOR_PROMPT.format(agents)
        self.clear_history()

        self.agent_messages = ""
        self._iterative_mode = True

    def transfer_to_agent(self, role: str, task: str) -> str:
        """Delegate a task to an agent based on its role."""
        with self._tracer.start_as_current_span("invoke_agent") as span:
            agent = self._agents.get(role.lower())
            if not agent:
                return f"No agent found with role '{role}'."

            span.add_event("Calling agent")
            span.set_attribute("agent.role", agent.role)
            agent_response = agent(task)
            span.set_attribute("agent.response", agent_response)

            self.agent_messages += agent_response

            span.set_status(Status(StatusCode.OK))
            return agent_response
