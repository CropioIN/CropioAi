from typing import List

from pydantic import BaseModel, Field


class ChatInputField(BaseModel):
    """
    Represents a single required input for the cropio, with a name and short description.
    Example:
        {
            "name": "topic",
            "description": "The topic to focus on for the conversation"
        }
    """

    name: str = Field(..., description="The name of the input field")
    description: str = Field(..., description="A short description of the input field")


class ChatInputs(BaseModel):
    """
    Holds a high-level cropio_description plus a list of ChatInputFields.
    Example:
        {
            "cropio_name": "topic-based-qa",
            "cropio_description": "Use this cropio for topic-based Q&A",
            "inputs": [
                {"name": "topic", "description": "The topic to focus on"},
                {"name": "username", "description": "Name of the user"},
            ]
        }
    """

    cropio_name: str = Field(..., description="The name of the cropio")
    cropio_description: str = Field(
        ..., description="A description of the cropio's purpose"
    )
    inputs: List[ChatInputField] = Field(
        default_factory=list, description="A list of input fields for the cropio"
    )
