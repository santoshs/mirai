"""Utility functions for agents and manager."""

import inspect
from typing import Callable, Dict, Any


def get_function_info(func: Callable) -> Dict[str, Any]:
    """Extract metadata for a given function to structure the prompt.

    Args:
        func (Callable): The function to extract information from.

    Returns:
        Dict[str, Any]: A dictionary containing function metadata.
    """
    sig = inspect.signature(func)
    params = {
        k: {
            "type": get_param_type(v.annotation),
            "description": "",
        }
        for k, v in sig.parameters.items()
    }
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": params,
                "required": list(params.keys()),
            },
        }
    }


def get_param_type(annotation) -> str:
    """Map Python types to JSON schema types.

    Args:
        annotation: The annotation to map.

    Returns:
        str: The corresponding JSON schema type.
    """
    if annotation == int:
        return "integer"
    elif annotation == float:
        return "number"
    elif annotation == bool:
        return "boolean"
    elif annotation == str:
        return "string"
    elif annotation == dict:
        return "object"
    elif annotation == list:
        return "array"
    return "string"
