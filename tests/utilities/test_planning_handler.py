from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from cropioai.agent import Agent
from cropioai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from cropioai.task import Task
from cropioai.tasks.task_output import TaskOutput
from cropioai.tools.base_tool import BaseTool
from cropioai.utilities.planning_handler import (
    CropioPlanner,
    PlannerTaskPydanticOutput,
    PlanPerTask,
)


class InternalCropioPlanner:
    @pytest.fixture
    def cropio_planner(self):
        tasks = [
            Task(
                description="Task 1",
                expected_output="Output 1",
                agent=Agent(role="Agent 1", goal="Goal 1", backstory="Backstory 1"),
            ),
            Task(
                description="Task 2",
                expected_output="Output 2",
                agent=Agent(role="Agent 2", goal="Goal 2", backstory="Backstory 2"),
            ),
            Task(
                description="Task 3",
                expected_output="Output 3",
                agent=Agent(role="Agent 3", goal="Goal 3", backstory="Backstory 3"),
            ),
        ]
        return CropioPlanner(tasks, None)

    @pytest.fixture
    def cropio_planner_different_llm(self):
        tasks = [
            Task(
                description="Task 1",
                expected_output="Output 1",
                agent=Agent(role="Agent 1", goal="Goal 1", backstory="Backstory 1"),
            )
        ]
        planning_agent_llm = "gpt-3.5-turbo"
        return CropioPlanner(tasks, planning_agent_llm)

    def test_handle_cropio_planning(self, cropio_planner):
        list_of_plans_per_task = [
            PlanPerTask(task="Task1", plan="Plan 1"),
            PlanPerTask(task="Task2", plan="Plan 2"),
            PlanPerTask(task="Task3", plan="Plan 3"),
        ]
        with patch.object(Task, "execute_sync") as execute:
            execute.return_value = TaskOutput(
                description="Description",
                agent="agent",
                pydantic=PlannerTaskPydanticOutput(
                    list_of_plans_per_task=list_of_plans_per_task
                ),
            )
            result = cropio_planner._handle_cropio_planning()
            assert cropio_planner.planning_agent_llm == "gpt-4o-mini"
            assert isinstance(result, PlannerTaskPydanticOutput)
            assert len(result.list_of_plans_per_task) == len(cropio_planner.tasks)
            execute.assert_called_once()

    def test_create_planning_agent(self, cropio_planner):
        agent = cropio_planner._create_planning_agent()
        assert isinstance(agent, Agent)
        assert agent.role == "Task Execution Planner"

    def test_create_planner_task(self, cropio_planner):
        planning_agent = Agent(
            role="Planning Agent",
            goal="Plan Step by Step Plan",
            backstory="Master in Planning",
        )
        tasks_summary = "Summary of tasks"
        task = cropio_planner._create_planner_task(planning_agent, tasks_summary)

        assert isinstance(task, Task)
        assert task.description.startswith("Based on these tasks summary")
        assert task.agent == planning_agent
        assert (
            task.expected_output
            == "Step by step plan on how the agents can execute their tasks using the available tools with mastery"
        )

    def test_create_tasks_summary(self, cropio_planner):
        tasks_summary = cropio_planner._create_tasks_summary()
        assert isinstance(tasks_summary, str)
        assert tasks_summary.startswith("\n                Task Number 1 - Task 1")
        assert '"agent_tools": "agent has no tools"' in tasks_summary
        # Knowledge field should not be present when empty
        assert '"agent_knowledge"' not in tasks_summary

    @patch('cropioai.knowledge.storage.knowledge_storage.chromadb')
    def test_create_tasks_summary_with_knowledge_and_tools(self, mock_chroma):
        """Test task summary generation with both knowledge and tools present."""
        # Mock ChromaDB collection
        mock_collection = mock_chroma.return_value.get_or_create_collection.return_value
        mock_collection.add.return_value = None

        # Create mock tools with proper string descriptions and structured tool support
        class MockTool(BaseTool):
            name: str
            description: str

            def __init__(self, name: str, description: str):
                tool_data = {"name": name, "description": description}
                super().__init__(**tool_data)

            def __str__(self):
                return self.name

            def __repr__(self):
                return self.name

            def to_structured_tool(self):
                return self

            def _run(self, *args, **kwargs):
                pass

            def _generate_description(self) -> str:
                """Override _generate_description to avoid args_schema handling."""
                return self.description

        tool1 = MockTool("tool1", "Tool 1 description")
        tool2 = MockTool("tool2", "Tool 2 description")

        # Create a task with knowledge and tools
        task = Task(
            description="Task with knowledge and tools",
            expected_output="Expected output",
            agent=Agent(
                role="Test Agent",
                goal="Test Goal",
                backstory="Test Backstory",
                tools=[tool1, tool2],
                knowledge_sources=[
                    StringKnowledgeSource(content="Test knowledge content")
                ]
            )
        )

        # Create planner with the new task
        planner = CropioPlanner([task], None)
        tasks_summary = planner._create_tasks_summary()

        # Verify task summary content
        assert isinstance(tasks_summary, str)
        assert task.description in tasks_summary
        assert task.expected_output in tasks_summary
        assert '"agent_tools": [tool1, tool2]' in tasks_summary
        assert '"agent_knowledge": "[\\"Test knowledge content\\"]"' in tasks_summary
        assert task.agent.role in tasks_summary
        assert task.agent.goal in tasks_summary

    def test_handle_cropio_planning_different_llm(self, cropio_planner_different_llm):
        with patch.object(Task, "execute_sync") as execute:
            execute.return_value = TaskOutput(
                description="Description",
                agent="agent",
                pydantic=PlannerTaskPydanticOutput(
                    list_of_plans_per_task=[PlanPerTask(task="Task1", plan="Plan 1")]
                ),
            )
            result = cropio_planner_different_llm._handle_cropio_planning()

            assert cropio_planner_different_llm.planning_agent_llm == "gpt-3.5-turbo"
            assert isinstance(result, PlannerTaskPydanticOutput)
            assert len(result.list_of_plans_per_task) == len(
                cropio_planner_different_llm.tasks
            )
            execute.assert_called_once()
