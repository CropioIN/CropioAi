from cropioai import Agent, Cropio, Process, Task
from cropioai.project import CropioBase, agent, cropio, task

# If you want to run a snippet of code before or after the cropio starts,
# you can use the @before_takeoff and @after_takeoff decorators
# https://docs.cropio.in/concepts/cropios#example-cropio-class-with-decorators


@CropioBase
class PoemCropio:
    """Poem Cropio"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.cropio.in/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.cropio.in/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # If you would lik to add tools to your cropio, you can learn more about it here:
    # https://docs.cropio.in/concepts/agents#agent-tools
    @agent
    def poem_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["poem_writer"],
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.cropio.in/concepts/tasks#overview-of-a-task
    @task
    def write_poem(self) -> Task:
        return Task(
            config=self.tasks_config["write_poem"],
        )

    @cropio
    def cropio(self) -> Cropio:
        """Creates the Research Cropio"""
        # To learn how to add knowledge sources to your cropio, check out the documentation:
        # https://docs.cropio.in/concepts/knowledge#what-is-knowledge

        return Cropio(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
