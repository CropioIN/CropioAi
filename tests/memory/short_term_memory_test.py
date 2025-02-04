from unittest.mock import patch

import pytest

from cropioai.agent import Agent
from cropioai.cropio import Cropio
from cropioai.memory.short_term.short_term_memory import ShortTermMemory
from cropioai.memory.short_term.short_term_memory_item import ShortTermMemoryItem
from cropioai.task import Task


@pytest.fixture
def short_term_memory():
    """Fixture to create a ShortTermMemory instance"""
    agent = Agent(
        role="Researcher",
        goal="Search relevant data and provide results",
        backstory="You are a researcher at a leading tech think tank.",
        tools=[],
        verbose=True,
    )

    task = Task(
        description="Perform a search on specific topics.",
        expected_output="A list of relevant URLs based on the search query.",
        agent=agent,
    )
    return ShortTermMemory(cropio=Cropio(agents=[agent], tasks=[task]))


def test_save_and_search(short_term_memory):
    memory = ShortTermMemoryItem(
        data="""test value test value test value test value test value test value
        test value test value test value test value test value test value
        test value test value test value test value test value test value""",
        agent="test_agent",
        metadata={"task": "test_task"},
    )

    with patch.object(ShortTermMemory, "save") as mock_save:
        short_term_memory.save(
            value=memory.data,
            metadata=memory.metadata,
            agent=memory.agent,
        )

        mock_save.assert_called_once_with(
            value=memory.data,
            metadata=memory.metadata,
            agent=memory.agent,
        )

    expected_result = [
        {
            "context": memory.data,
            "metadata": {"agent": "test_agent"},
            "score": 0.95,
        }
    ]
    with patch.object(ShortTermMemory, "search", return_value=expected_result):
        find = short_term_memory.search("test value", score_threshold=0.01)[0]
        assert find["context"] == memory.data, "Data value mismatch."
        assert find["metadata"]["agent"] == "test_agent", "Agent value mismatch."
