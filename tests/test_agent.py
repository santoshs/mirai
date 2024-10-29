import os
import unittest
from agent import Agent
from dotenv import load_dotenv


def greet(name: str) -> str:
    """Greet the person by name."""
    return f"Hiya {name}. Nor a drop to drink!"


class TestAgent(unittest.TestCase):

    def setUp(self):
        load_dotenv()
        api_key = os.getenv('NVIDIA_API_KEY')

        # Initialize an Agent instance for testing, base URL, and functions
        self.agent = Agent(
            role="test_role",
            role_description="A test agent for handling tasks.",
            functions=[greet],
            model="meta/llama-3.1-8b-instruct",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )

    def test_respond(self):
        user_input = "Hello"

        self.agent(user_input)

    def test_tool_calling(self):
        user_input = "Can you please greet Popeye?"

        self.agent(user_input)


if __name__ == "__main__":
    unittest.main()
