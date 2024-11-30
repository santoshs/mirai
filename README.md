Mirai is a Python-based framework designed to orchestrate multiple Large
Language Model (LLM) agents, enabling them to collaborate on complex tasks. It
provides a structured approach to define, manage, and execute tasks across
various agents, facilitating seamless integration and coordination.  Features

- Modular Design: Mirai's architecture allows for easy addition and management of diverse agents, each specializing in specific tasks.
- Flexible Orchestration: Define workflows where multiple agents can interact and collaborate to achieve complex objectives.
- Extensibility: Easily integrate new agents or tools to expand the system's capabilities.

## Installation

To install Mirai, clone the repository and install the required dependencies:

```bash
pip install mirai@git+https://github.com/santoshs/mirai
```

## Usage
Mirai enables the creation of agents with specific roles and functions. For
example, to define a web search agent:

```python
from mirai import Agent

search_agent = Agent(
    role="Web Search Agent",
    role_description="Performs web searches and retrieves information.",
    functions=[search_internet, scrape_web],
    api_key="your_api_key",
    model="your_model_name",
)

manager = AgentManager(
    agents=[search_agent],
    api_key=os.getenv("OPENAI_API_KEY"),
    model=MODEL,
)

manager("Who is Popeye the sailor?")
```

This agent can then be managed and orchestrated alongside other agents to
perform complex tasks.

For complex tasks, derive the Agent class and override the `__call__`
method. For very simple tasks which doesn't require tools or LLM, just pass a
function as the `run` paramter when instantiating the Agent class. The function
should accept a string and return the string.
