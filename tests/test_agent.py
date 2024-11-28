import os
import unittest
from agent import Agent
from dotenv import load_dotenv
import logging
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register


logging.basicConfig(filename="tests.log", level=logging.DEBUG)

tracer_provider = register(
    project_name="test",
    endpoint="http://localhost:6006/v1/traces",
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


class TestBasicRun(unittest.TestCase):

    def setUp(self):
        self.agent = Agent(
            role="A echo",
            role_description="I just echo what you say",
            model="",
            api_key="",
            run=self.echo,
        )

    def test_run(self):
        response = self.agent("Hello Bill")
        self.assertEqual(response, "Hello Bill")

    def echo(self, input_text):
        return input_text


class TestAgent(unittest.TestCase):

    def setUp(self):
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        self.log = logging.getLogger(__name__)

        # Initialize an Agent instance for testing, base URL, and functions
        self.agent = Agent(
            role="test_role",
            role_description="A test agent for handling tasks.",
            functions=[self.greet],
            model="gpt-4o-mini",
            api_key=api_key,
        )

    def test_response(self):
        user_input = "Hello"

        response = self.agent(user_input)
        self.log.debug(response)
        self.assertIsInstance(response, str)

    def greet(self, name: str) -> str:
        """Greet the person by name."""
        return f"Hiya {name}. Nor a drop to drink!"

    def test_tool_calling(self):
        user_input = "Can you please greet Popeye?"
        self.log.debug(user_input)

        response = self.agent(user_input)

        self.log.debug(response)
        self.assertRegex(response, "^Hiya Popeye.*Nor a drop to drink.*$")
