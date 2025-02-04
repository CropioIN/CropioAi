import time
from typing import TYPE_CHECKING, Optional

from cropioai.memory.entity.entity_memory_item import EntityMemoryItem
from cropioai.memory.long_term.long_term_memory_item import LongTermMemoryItem
from cropioai.utilities import I18N
from cropioai.utilities.converter import ConverterError
from cropioai.utilities.evaluators.task_evaluator import TaskEvaluator
from cropioai.utilities.printer import Printer

if TYPE_CHECKING:
    from cropioai.agents.agent_builder.base_agent import BaseAgent
    from cropioai.cropio import Cropio
    from cropioai.task import Task


class CropioAgentExecutorMixin:
    cropio: Optional["Cropio"]
    agent: Optional["BaseAgent"]
    task: Optional["Task"]
    iterations: int
    max_iter: int
    _i18n: I18N
    _printer: Printer = Printer()

    def _create_short_term_memory(self, output) -> None:
        """Create and save a short-term memory item if conditions are met."""
        if (
            self.cropio
            and self.agent
            and self.task
            and "Action: Delegate work to coworker" not in output.text
        ):
            try:
                if (
                    hasattr(self.cropio, "_short_term_memory")
                    and self.cropio._short_term_memory
                ):
                    self.cropio._short_term_memory.save(
                        value=output.text,
                        metadata={
                            "observation": self.task.description,
                        },
                        agent=self.agent.role,
                    )
            except Exception as e:
                print(f"Failed to add to short term memory: {e}")
                pass

    def _create_long_term_memory(self, output) -> None:
        """Create and save long-term and entity memory items based on evaluation."""
        if (
            self.cropio
            and self.cropio.memory
            and self.cropio._long_term_memory
            and self.cropio._entity_memory
            and self.task
            and self.agent
        ):
            try:
                ltm_agent = TaskEvaluator(self.agent)
                evaluation = ltm_agent.evaluate(self.task, output.text)

                if isinstance(evaluation, ConverterError):
                    return

                long_term_memory = LongTermMemoryItem(
                    task=self.task.description,
                    agent=self.agent.role,
                    quality=evaluation.quality,
                    datetime=str(time.time()),
                    expected_output=self.task.expected_output,
                    metadata={
                        "suggestions": evaluation.suggestions,
                        "quality": evaluation.quality,
                    },
                )
                self.cropio._long_term_memory.save(long_term_memory)

                for entity in evaluation.entities:
                    entity_memory = EntityMemoryItem(
                        name=entity.name,
                        type=entity.type,
                        description=entity.description,
                        relationships="\n".join(
                            [f"- {r}" for r in entity.relationships]
                        ),
                    )
                    self.cropio._entity_memory.save(entity_memory)
            except AttributeError as e:
                print(f"Missing attributes for long term memory: {e}")
                pass
            except Exception as e:
                print(f"Failed to add to long term memory: {e}")
                pass

    def _ask_human_input(self, final_answer: str) -> str:
        """Prompt human input with mode-appropriate messaging."""
        self._printer.print(
            content=f"\033[1m\033[95m ## Final Result:\033[00m \033[92m{final_answer}\033[00m"
        )

        # Training mode prompt (single iteration)
        if self.cropio and getattr(self.cropio, "_train", False):
            prompt = (
                "\n\n=====\n"
                "## TRAINING MODE: Provide feedback to improve the agent's performance.\n"
                "This will be used to train better versions of the agent.\n"
                "Please provide detailed feedback about the result quality and reasoning process.\n"
                "=====\n"
            )
        # Regular human-in-the-loop prompt (multiple iterations)
        else:
            prompt = (
                "\n\n=====\n"
                "## HUMAN FEEDBACK: Provide feedback on the Final Result and Agent's actions.\n"
                "Respond with 'looks good' to accept or provide specific improvement requests.\n"
                "You can provide multiple rounds of feedback until satisfied.\n"
                "=====\n"
            )

        self._printer.print(content=prompt, color="bold_yellow")
        return input()
