from cropioai import Agent, Cropio, Process, Task
from cropioai.project import CropioBase, agent, cropio, task

# If you want to run a snippet of code before or after the cropio starts, 
# you can use the @before_takeoff and @after_takeoff decorators
# https://docs.cropio.in/concepts/cropios#example-cropio-class-with-decorators

@CropioBase
class {{cropio_name}}():
	"""{{cropio_name}} cropio"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.cropio.in/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.cropio.in/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.cropio.in/concepts/agents#agent-tools
	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			verbose=True
		)

	@agent
	def reporting_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['reporting_analyst'],
			verbose=True
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.cropio.in/concepts/tasks#overview-of-a-task
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
		)

	@task
	def reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task'],
			output_file='report.md'
		)

	@cropio
	def cropio(self) -> Cropio:
		"""Creates the {{cropio_name}} cropio"""
		# To learn how to add knowledge sources to your cropio, check out the documentation:
		# https://docs.cropio.in/concepts/knowledge#what-is-knowledge

		return Cropio(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.cropio.in/how-to/Hierarchical/
		)
