from unittest import mock

import pytest

from cropioai.agent import Agent
from cropioai.cropio import Cropio
from cropioai.task import Task
from cropioai.tasks.task_output import TaskOutput
from cropioai.utilities.evaluators.cropio_evaluator_handler import (
    CropioEvaluator,
    TaskEvaluationPydanticOutput,
)


class InternalCropioEvaluator:
    @pytest.fixture
    def cropio_planner(self):
        agent = Agent(role="Agent 1", goal="Goal 1", backstory="Backstory 1")
        task = Task(
            description="Task 1",
            expected_output="Output 1",
            agent=agent,
        )
        cropio = Cropio(agents=[agent], tasks=[task])

        return CropioEvaluator(cropio, openai_model_name="gpt-4o-mini")

    def test_setup_for_evaluating(self, cropio_planner):
        cropio_planner._setup_for_evaluating()
        assert cropio_planner.cropio.tasks[0].callback == cropio_planner.evaluate

    def test_set_iteration(self, cropio_planner):
        cropio_planner.set_iteration(1)
        assert cropio_planner.iteration == 1

    def test_evaluator_agent(self, cropio_planner):
        agent = cropio_planner._evaluator_agent()
        assert agent.role == "Task Execution Evaluator"
        assert (
            agent.goal
            == "Your goal is to evaluate the performance of the agents in the cropio based on the tasks they have performed using score from 1 to 10 evaluating on completion, quality, and overall performance."
        )
        assert (
            agent.backstory
            == "Evaluator agent for cropio evaluation with precise capabilities to evaluate the performance of the agents in the cropio based on the tasks they have performed"
        )
        assert agent.verbose is False
        assert agent.llm.model == "gpt-4o-mini"

    def test_evaluation_task(self, cropio_planner):
        evaluator_agent = Agent(
            role="Evaluator Agent",
            goal="Evaluate the performance of the agents in the cropio",
            backstory="Master in Evaluation",
        )
        task_to_evaluate = Task(
            description="Task 1",
            expected_output="Output 1",
            agent=Agent(role="Agent 1", goal="Goal 1", backstory="Backstory 1"),
        )
        task_output = "Task Output 1"
        task = cropio_planner._evaluation_task(
            evaluator_agent, task_to_evaluate, task_output
        )

        assert task.description.startswith(
            "Based on the task description and the expected output, compare and evaluate the performance of the agents in the cropio based on the Task Output they have performed using score from 1 to 10 evaluating on completion, quality, and overall performance."
        )

        assert task.agent == evaluator_agent
        assert (
            task.description
            == "Based on the task description and the expected output, compare and evaluate "
            "the performance of the agents in the cropio based on the Task Output they have "
            "performed using score from 1 to 10 evaluating on completion, quality, and overall "
            "performance.task_description: Task 1 task_expected_output: Output 1 "
            "agent: Agent 1 agent_goal: Goal 1 Task Output: Task Output 1"
        )

    @mock.patch("cropioai.utilities.evaluators.cropio_evaluator_handler.Console")
    @mock.patch("cropioai.utilities.evaluators.cropio_evaluator_handler.Table")
    def test_print_cropio_evaluation_result(self, table, console, cropio_planner):
        # Set up task scores and execution times
        cropio_planner.tasks_scores = {
            1: [10, 9, 8],
            2: [9, 8, 7],
        }
        cropio_planner.run_execution_times = {
            1: [24, 45, 66],
            2: [55, 33, 67],
        }

        # Mock agents and assign them to tasks
        cropio_planner.cropio.agents = [
            mock.Mock(role="Agent 1"),
            mock.Mock(role="Agent 2"),
        ]
        cropio_planner.cropio.tasks = [
            mock.Mock(
                agent=cropio_planner.cropio.agents[0], processed_by_agents=["Agent 1"]
            ),
            mock.Mock(
                agent=cropio_planner.cropio.agents[1], processed_by_agents=["Agent 2"]
            ),
        ]

        # Run the method
        cropio_planner.print_cropio_evaluation_result()

        # Verify that the table is created with the appropriate structure and rows
        table.assert_has_calls(
            [
                mock.call(
                    title="Tasks Scores \n (1-10 Higher is better)", box=mock.ANY
                ),  # Title and styling
                mock.call().add_column("Tasks/Cropio/Agents", style="cyan"),  # Columns
                mock.call().add_column("Run 1", justify="center"),
                mock.call().add_column("Run 2", justify="center"),
                mock.call().add_column("Avg. Total", justify="center"),
                mock.call().add_column("Agents", style="green"),
                # Verify rows for tasks with agents
                mock.call().add_row("Task 1", "10.0", "9.0", "9.5", "- Agent 1"),
                mock.call().add_row("", "", "", "", "", ""),  # Blank row between tasks
                mock.call().add_row("Task 2", "9.0", "8.0", "8.5", "- Agent 2"),
                # Add cropio averages and execution times
                mock.call().add_row("Cropio", "9.00", "8.00", "8.5", ""),
                mock.call().add_row("Execution Time (s)", "135", "155", "145", ""),
            ]
        )

        # Ensure the console prints the table
        console.assert_has_calls([mock.call(), mock.call().print(table())])

    def test_evaluate(self, cropio_planner):
        task_output = TaskOutput(
            description="Task 1", agent=str(cropio_planner.cropio.agents[0])
        )

        with mock.patch.object(Task, "execute_sync") as execute:
            execute().pydantic = TaskEvaluationPydanticOutput(quality=9.5)
            cropio_planner.evaluate(task_output)
            assert cropio_planner.tasks_scores[0] == [9.5]
