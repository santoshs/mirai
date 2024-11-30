"""Package setup file."""

from setuptools import setup, find_packages

setup(
    name="mirai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add any dependencies your framework requires
        "opentelemetry-semantic-conventions",
        "openai",
        "openinference-instrumentation-openai",
        "python-dotenv",
        "arize-phoenix"
    ],
    include_package_data=True,
)
