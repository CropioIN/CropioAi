<div align="center">

![Logo of CropioAI, two people rowing on a boat](https://raw.githubusercontent.com/cropioin/cropioai/main/docs/Cropio.gif)

# **CropioAI**

🤖 **CropioAI**: Production-grade framework for orchestrating sophisticated AI agent systems. From simple automations to complex real-world applications, CropioAI provides precise control and deep customization. By fostering collaborative intelligence through flexible, production-ready architecture, CropioAI empowers agents to work together seamlessly, tackling complex business challenges with predictable, consistent results.

<h3>

[Homepage](https://www.cropio.in/) | [Documentation](https://docs.cropio.in/) | [Chat with Docs](https://chatg.pt/DWjSBZn) | [Examples](https://github.com/cropioin/cropioAI-examples) | [Discourse](https://community.cropio.in)

</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/cropioin/cropioAI)](https://github.com/cropioin/cropioAI)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

## Table of contents

- [Why CropioAI?](#why-cropioai)
- [Getting Started](#getting-started)
- [Key Features](#key-features)
- [Understanding Flows and Cropios](#understanding-flows-and-cropios)
- [CropioAI vs LangGraph](#how-cropioai-compares)
- [Examples](#examples)
  - [Quick Tutorial](#quick-tutorial)
  - [Write Job Descriptions](#write-job-descriptions)
  - [Trip Planner](#trip-planner)
  - [Stock Analysis](#stock-analysis)
  - [Using Cropios and Flows Together](#using-cropios-and-flows-together)
- [Connecting Your Cropio to a Model](#connecting-your-cropio-to-a-model)
- [How CropioAI Compares](#how-cropioai-compares)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
- [Contribution](#contribution)
- [Telemetry](#telemetry)
- [License](#license)

## Why CropioAI?

The power of AI collaboration has too much to offer.
CropioAI is a standalone framework, built from the ground up without dependencies on Langchain or other agent frameworks. It's designed to enable AI agents to assume roles, share goals, and operate in a cohesive unit - much like a well-oiled cropio. Whether you're building a smart assistant platform, an automated customer service ensemble, or a multi-agent research team, CropioAI provides the backbone for sophisticated multi-agent interactions.

## Getting Started

### Learning Resources

Learn CropioAI through our comprehensive courses:
- [Multi AI Agent Systems with CropioAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-cropioai/) - Master the fundamentals of multi-agent systems
- [Practical Multi AI Agents and Advanced Use Cases](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-cropioai/) - Deep dive into advanced implementations

### Understanding Flows and Cropios

CropioAI offers two powerful, complementary approaches that work seamlessly together to build sophisticated AI applications:

1. **Cropios**: Teams of AI agents with true autonomy and agency, working together to accomplish complex tasks through role-based collaboration. Cropios enable:
   - Natural, autonomous decision-making between agents
   - Dynamic task delegation and collaboration
   - Specialized roles with defined goals and expertise
   - Flexible problem-solving approaches

2. **Flows**: Production-ready, event-driven workflows that deliver precise control over complex automations. Flows provide:
   - Fine-grained control over execution paths for real-world scenarios
   - Secure, consistent state management between tasks
   - Clean integration of AI agents with production Python code
   - Conditional branching for complex business logic

The true power of CropioAI emerges when combining Cropios and Flows. This synergy allows you to:
- Build complex, production-grade applications
- Balance autonomy with precise control
- Handle sophisticated real-world scenarios
- Maintain clean, maintainable code structure

### Getting Started with Installation

To get started with CropioAI, follow these simple steps:

### 1. Installation

Ensure you have Python >=3.10 <3.13 installed on your system. CropioAI uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, install CropioAI:

```shell
pip install cropioai
```
If you want to install the 'cropioai' package along with its optional features that include additional tools for agents, you can do so by using the following command:

```shell
pip install 'cropioai[tools]'
```
The command above installs the basic package and also adds extra components which require more dependencies to function.

### Troubleshooting Dependencies

If you encounter issues during installation or usage, here are some common solutions:

#### Common Issues

1. **ModuleNotFoundError: No module named 'tiktoken'**
   - Install tiktoken explicitly: `pip install 'cropioai[embeddings]'`
   - If using embedchain or other tools: `pip install 'cropioai[tools]'`

2. **Failed building wheel for tiktoken**
   - Ensure Rust compiler is installed (see installation steps above)
   - For Windows: Verify Visual C++ Build Tools are installed
   - Try upgrading pip: `pip install --upgrade pip`
   - If issues persist, use a pre-built wheel: `pip install tiktoken --prefer-binary`

### 2. Setting Up Your Cropio with the YAML Configuration

To create a new CropioAI project, run the following CLI (Command Line Interface) command:

```shell
cropioai create cropio <project_name>
```

This command creates a new project folder with the following structure:

```
my_project/
├── .gitignore
├── pyproject.toml
├── README.md
├── .env
└── src/
    └── my_project/
        ├── __init__.py
        ├── main.py
        ├── cropio.py
        ├── tools/
        │   ├── custom_tool.py
        │   └── __init__.py
        └── config/
            ├── agents.yaml
            └── tasks.yaml
```

You can now start developing your cropio by editing the files in the `src/my_project` folder. The `main.py` file is the entry point of the project, the `cropio.py` file is where you define your cropio, the `agents.yaml` file is where you define your agents, and the `tasks.yaml` file is where you define your tasks.

#### To customize your project, you can:

- Modify `src/my_project/config/agents.yaml` to define your agents.
- Modify `src/my_project/config/tasks.yaml` to define your tasks.
- Modify `src/my_project/cropio.py` to add your own logic, tools, and specific arguments.
- Modify `src/my_project/main.py` to add custom inputs for your agents and tasks.
- Add your environment variables into the `.env` file.

#### Example of a simple cropio with a sequential process:

Instantiate your cropio:

```shell
cropioai create cropio latest-ai-development
```

Modify the files as needed to fit your use case:

**agents.yaml**

```yaml
# src/my_project/config/agents.yaml
researcher:
  role: >
    {topic} Senior Data Researcher
  goal: >
    Uncover cutting-edge developments in {topic}
  backstory: >
    You're a seasoned researcher with a knack for uncovering the latest
    developments in {topic}. Known for your ability to find the most relevant
    information and present it in a clear and concise manner.

reporting_analyst:
  role: >
    {topic} Reporting Analyst
  goal: >
    Create detailed reports based on {topic} data analysis and research findings
  backstory: >
    You're a meticulous analyst with a keen eye for detail. You're known for
    your ability to turn complex data into clear and concise reports, making
    it easy for others to understand and act on the information you provide.
```

**tasks.yaml**

```yaml
# src/my_project/config/tasks.yaml
research_task:
  description: >
    Conduct a thorough research about {topic}
    Make sure you find any interesting and relevant information given
    the current year is 2024.
  expected_output: >
    A list with 10 bullet points of the most relevant information about {topic}
  agent: researcher

reporting_task:
  description: >
    Review the context you got and expand each topic into a full section for a report.
    Make sure the report is detailed and contains any and all relevant information.
  expected_output: >
    A fully fledge reports with the mains topics, each with a full section of information.
    Formatted as markdown without '```'
  agent: reporting_analyst
  output_file: report.md
```

**cropio.py**

```python
# src/my_project/cropio.py
from cropioai import Agent, Cropio, Process, Task
from cropioai.project import CropioBase, agent, cropio, task
from cropioai_tools import SerperDevTool

@CropioBase
class LatestAiDevelopmentCropio():
	"""LatestAiDevelopment cropio"""

	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			verbose=True,
			tools=[SerperDevTool()]
		)

	@agent
	def reporting_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['reporting_analyst'],
			verbose=True
		)

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
		"""Creates the LatestAiDevelopment cropio"""
		return Cropio(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)
```

**main.py**

```python
#!/usr/bin/env python
# src/my_project/main.py
import sys
from latest_ai_development.cropio import LatestAiDevelopmentCropio

def run():
    """
    Run the cropio.
    """
    inputs = {
        'topic': 'AI Agents'
    }
    LatestAiDevelopmentCropio().cropio().takeoff(inputs=inputs)
```

### 3. Running Your Cropio

Before running your cropio, make sure you have the following keys set as environment variables in your `.env` file:

- An [OpenAI API key](https://platform.openai.com/account/api-keys) (or other LLM API key): `OPENAI_API_KEY=sk-...`
- A [Serper.dev](https://serper.dev/) API key: `SERPER_API_KEY=YOUR_KEY_HERE`

Lock the dependencies and install them by using the CLI command but first, navigate to your project directory:

```shell
cd my_project
cropioai install (Optional)
```

To run your cropio, execute the following command in the root of your project:

```bash
cropioai run
```

or

```bash
python src/my_project/main.py
```

If an error happens due to the usage of poetry, please run the following command to update your cropioai package:

```bash
cropioai update
```

You should see the output in the console and the `report.md` file should be created in the root of your project with the full final report.

In addition to the sequential process, you can use the hierarchical process, which automatically assigns a manager to the defined cropio to properly coordinate the planning and execution of tasks through delegation and validation of results. [See more about the processes here](https://docs.cropio.in/core-concepts/Processes/).

## Key Features

**Note**: CropioAI is a standalone framework built from the ground up, without dependencies on Langchain or other agent frameworks.

- **Deep Customization**: Build sophisticated agents with full control over the system - from overriding inner prompts to accessing low-level APIs. Customize roles, goals, tools, and behaviors while maintaining clean abstractions.
- **Autonomous Inter-Agent Delegation**: Agents can autonomously delegate tasks and inquire amongst themselves, enabling complex problem-solving in real-world scenarios.
- **Flexible Task Management**: Define and customize tasks with granular control, from simple operations to complex multi-step processes.
- **Production-Grade Architecture**: Support for both high-level abstractions and low-level customization, with robust error handling and state management.
- **Predictable Results**: Ensure consistent, accurate outputs through programmatic guardrails, agent training capabilities, and flow-based execution control. See our [documentation on guardrails](https://docs.cropio.in/how-to/guardrails/) for implementation details.
- **Model Flexibility**: Run your cropio using OpenAI or open source models with production-ready integrations. See [Connect CropioAI to LLMs](https://docs.cropio.in/how-to/LLM-Connections/) for detailed configuration options.
- **Event-Driven Flows**: Build complex, real-world workflows with precise control over execution paths, state management, and conditional logic.
- **Process Orchestration**: Achieve any workflow pattern through flows - from simple sequential and hierarchical processes to complex, custom orchestration patterns with conditional branching and parallel execution.

## Examples

You can test different real life examples of AI cropios in the [CropioAI-examples repo](https://github.com/cropioin/cropioAI-examples?tab=readme-ov-file):

- [Landing Page Generator](https://github.com/cropioin/cropioAI-examples/tree/main/landing_page_generator)
- [Having Human input on the execution](https://docs.cropio.in/how-to/Human-Input-on-Execution)
- [Trip Planner](https://github.com/cropioin/cropioAI-examples/tree/main/trip_planner)
- [Stock Analysis](https://github.com/cropioin/cropioAI-examples/tree/main/stock_analysis)


### Using Cropios and Flows Together

CropioAI's power truly shines when combining Cropios with Flows to create sophisticated automation pipelines. Here's how you can orchestrate multiple Cropios within a Flow:

```python
from cropioai.flow.flow import Flow, listen, start, router
from cropioai import Cropio, Agent, Task
from pydantic import BaseModel

# Define structured state for precise control
class MarketState(BaseModel):
    sentiment: str = "neutral"
    confidence: float = 0.0
    recommendations: list = []

class AdvancedAnalysisFlow(Flow[MarketState]):
    @start()
    def fetch_market_data(self):
        # Demonstrate low-level control with structured state
        self.state.sentiment = "analyzing"
        return {"sector": "tech", "timeframe": "1W"}  # These parameters match the task description template

    @listen(fetch_market_data)
    def analyze_with_cropio(self, market_data):
        # Show cropio agency through specialized roles
        analyst = Agent(
            role="Senior Market Analyst",
            goal="Conduct deep market analysis with expert insight",
            backstory="You're a veteran analyst known for identifying subtle market patterns"
        )
        researcher = Agent(
            role="Data Researcher",
            goal="Gather and validate supporting market data",
            backstory="You excel at finding and correlating multiple data sources"
        )
        
        analysis_task = Task(
            description="Analyze {sector} sector data for the past {timeframe}",
            expected_output="Detailed market analysis with confidence score",
            agent=analyst
        )
        research_task = Task(
            description="Find supporting data to validate the analysis",
            expected_output="Corroborating evidence and potential contradictions",
            agent=researcher
        )
        
        # Demonstrate cropio autonomy
        analysis_cropio = Cropio(
            agents=[analyst, researcher],
            tasks=[analysis_task, research_task],
            process=Process.sequential,
            verbose=True
        )
        return analysis_cropio.takeoff(inputs=market_data)  # Pass market_data as named inputs

    @router(analyze_with_cropio)
    def determine_next_steps(self):
        # Show flow control with conditional routing
        if self.state.confidence > 0.8:
            return "high_confidence"
        elif self.state.confidence > 0.5:
            return "medium_confidence"
        return "low_confidence"

    @listen("high_confidence")
    def execute_strategy(self):
        # Demonstrate complex decision making
        strategy_cropio = Cropio(
            agents=[
                Agent(role="Strategy Expert",
                      goal="Develop optimal market strategy")
            ],
            tasks=[
                Task(description="Create detailed strategy based on analysis",
                     expected_output="Step-by-step action plan")
            ]
        )
        return strategy_cropio.takeoff()

    @listen("medium_confidence", "low_confidence")
    def request_additional_analysis(self):
        self.state.recommendations.append("Gather more data")
        return "Additional analysis required"
```

This example demonstrates how to:
1. Use Python code for basic data operations
2. Create and execute Cropios as steps in your workflow
3. Use Flow decorators to manage the sequence of operations
4. Implement conditional branching based on Cropio results

## Connecting Your Cropio to a Model

CropioAI supports using various LLMs through a variety of connection options. By default your agents will use the OpenAI API when querying the model. However, there are several other ways to allow your agents to connect to models. For example, you can configure your agents to use a local model via the Ollama tool.

Please refer to the [Connect CropioAI to LLMs](https://docs.cropio.in/how-to/LLM-Connections/) page for details on configuring you agents' connections to models.

## How CropioAI Compares

**CropioAI's Advantage**: CropioAI combines autonomous agent intelligence with precise workflow control through its unique Cropios and Flows architecture. The framework excels at both high-level orchestration and low-level customization, enabling complex, production-grade systems with granular control.

- **LangGraph**: While LangGraph provides a foundation for building agent workflows, its approach requires significant boilerplate code and complex state management patterns. The framework's tight coupling with LangChain can limit flexibility when implementing custom agent behaviors or integrating with external systems.

*P.S. CropioAI demonstrates significant performance advantages over LangGraph, executing 5.76x faster in certain cases like this QA task example ([see comparison](https://github.com/cropioin/cropioAI-examples/tree/main/Notebooks/CropioAI%20Flows%20%26%20Langgraph/QA%20Agent)) while achieving higher evaluation scores with faster completion times in certain coding tasks, like in this example ([detailed analysis](https://github.com/cropioin/cropioAI-examples/blob/main/Notebooks/CropioAI%20Flows%20%26%20Langgraph/Coding%20Assistant/coding_assistant_eval.ipynb)).*

- **Autogen**: While Autogen excels at creating conversational agents capable of working together, it lacks an inherent concept of process. In Autogen, orchestrating agents' interactions requires additional programming, which can become complex and cumbersome as the scale of tasks grows.

- **ChatDev**: ChatDev introduced the idea of processes into the realm of AI agents, but its implementation is quite rigid. Customizations in ChatDev are limited and not geared towards production environments, which can hinder scalability and flexibility in real-world applications.

## Contribution

CropioAI is open-source and we welcome contributions. If you're looking to contribute, please:

- Fork the repository.
- Create a new branch for your feature.
- Add your feature or improvement.
- Send a pull request.
- We appreciate your input!

### Installing Dependencies

```bash
uv lock
uv sync
```

### Virtual Env

```bash
uv venv
```

### Pre-commit hooks

```bash
pre-commit install
```

### Running Tests

```bash
uv run pytest .
```

### Running static type checks

```bash
uvx mypy src
```

### Packaging

```bash
uv build
```

### Installing Locally

```bash
pip install dist/*.tar.gz
```

## Telemetry

CropioAI uses anonymous telemetry to collect usage data with the main purpose of helping us improve the library by focusing our efforts on the most used features, integrations and tools.

It's pivotal to understand that **NO data is collected** concerning prompts, task descriptions, agents' backstories or goals, usage of tools, API calls, responses, any data processed by the agents, or secrets and environment variables, with the exception of the conditions mentioned. When the `share_cropio` feature is enabled, detailed data including task descriptions, agents' backstories or goals, and other specific attributes are collected to provide deeper insights while respecting user privacy. Users can disable telemetry by setting the environment variable OTEL_SDK_DISABLED to true.

Data collected includes:

- Version of CropioAI
  - So we can understand how many users are using the latest version
- Version of Python
  - So we can decide on what versions to better support
- General OS (e.g. number of CPUs, macOS/Windows/Linux)
  - So we know what OS we should focus on and if we could build specific OS related features
- Number of agents and tasks in a cropio
  - So we make sure we are testing internally with similar use cases and educate people on the best practices
- Cropio Process being used
  - Understand where we should focus our efforts
- If Agents are using memory or allowing delegation
  - Understand if we improved the features or maybe even drop them
- If Tasks are being executed in parallel or sequentially
  - Understand if we should focus more on parallel execution
- Language model being used
  - Improved support on most used languages
- Roles of agents in a cropio
  - Understand high level use cases so we can build better tools, integrations and examples about it
- Tools names available
  - Understand out of the publicly available tools, which ones are being used the most so we can improve them

Users can opt-in to Further Telemetry, sharing the complete telemetry data by setting the `share_cropio` attribute to `True` on their Cropios. Enabling `share_cropio` results in the collection of detailed cropio and task execution data, including `goal`, `backstory`, `context`, and `output` of tasks. This enables a deeper insight into usage patterns while respecting the user's choice to share.

## License

CropioAI is released under the [MIT License](https://github.com/cropioin/cropioAI/blob/main/LICENSE).

## Frequently Asked Questions (FAQ)

### Q: What is CropioAI?
A: CropioAI is a cutting-edge framework for orchestrating role-playing, autonomous AI agents. It enables agents to work together seamlessly, tackling complex tasks through collaborative intelligence.

### Q: How do I install CropioAI?
A: You can install CropioAI using pip:
```shell
pip install cropioai
```
For additional tools, use:
```shell
pip install 'cropioai[tools]'
```

### Q: Can I use CropioAI with local models?
A: Yes, CropioAI supports various LLMs, including local models. You can configure your agents to use local models via tools like Ollama & LM Studio. Check the [LLM Connections documentation](https://docs.cropio.in/how-to/LLM-Connections/) for more details.

### Q: What are the key features of CropioAI?
A: Key features include role-based agent design, autonomous inter-agent delegation, flexible task management, process-driven execution, output saving as files, and compatibility with both open-source and proprietary models.

### Q: How does CropioAI compare to other AI orchestration tools?
A: CropioAI is designed with production in mind, offering flexibility similar to Autogen's conversational agents and structured processes like ChatDev, but with more adaptability for real-world applications.

### Q: Is CropioAI open-source?
A: Yes, CropioAI is open-source and welcomes contributions from the community.

### Q: Does CropioAI collect any data?
A: CropioAI uses anonymous telemetry to collect usage data for improvement purposes. No sensitive data (like prompts, task descriptions, or API calls) is collected. Users can opt-in to share more detailed data by setting `share_cropio=True` on their Cropios.

### Q: Where can I find examples of CropioAI in action?
A: You can find various real-life examples in the [CropioAI-examples repository](https://github.com/cropioin/cropioAI-examples), including trip planners, stock analysis tools, and more.

### Q: What is the difference between Cropios and Flows?
A: Cropios and Flows serve different but complementary purposes in CropioAI. Cropios are teams of AI agents working together to accomplish specific tasks through role-based collaboration, delivering accurate and predictable results. Flows, on the other hand, are event-driven workflows that can orchestrate both Cropios and regular Python code, allowing you to build complex automation pipelines with secure state management and conditional execution paths.

### Q: How can I contribute to CropioAI?
A: Contributions are welcome! You can fork the repository, create a new branch for your feature, add your improvement, and send a pull request. Check the Contribution section in the README for more details.
