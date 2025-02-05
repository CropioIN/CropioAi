import subprocess

import click

from cropioai.knowledge.storage.knowledge_storage import KnowledgeStorage
from cropioai.memory.entity.entity_memory import EntityMemory
from cropioai.memory.long_term.long_term_memory import LongTermMemory
from cropioai.memory.short_term.short_term_memory import ShortTermMemory
from cropioai.utilities.task_output_storage_handler import TaskOutputStorageHandler


def reset_memories_command(
    long,
    short,
    entity,
    knowledge,
    takeoff_outputs,
    all,
) -> None:
    """
    Reset the cropio memories.

    Args:
      long (bool): Whether to reset the long-term memory.
      short (bool): Whether to reset the short-term memory.
      entity (bool): Whether to reset the entity memory.
      takeoff_outputs (bool): Whether to reset the latest takeoff task outputs.
      all (bool): Whether to reset all memories.
      knowledge (bool): Whether to reset the knowledge.
    """

    try:
        if all:
            ShortTermMemory().reset()
            EntityMemory().reset()
            LongTermMemory().reset()
            TaskOutputStorageHandler().reset()
            KnowledgeStorage().reset()
            click.echo("All memories have been reset.")
        else:
            if long:
                LongTermMemory().reset()
                click.echo("Long term memory has been reset.")

            if short:
                ShortTermMemory().reset()
                click.echo("Short term memory has been reset.")
            if entity:
                EntityMemory().reset()
                click.echo("Entity memory has been reset.")
            if takeoff_outputs:
                TaskOutputStorageHandler().reset()
                click.echo("Latest Takeoff outputs stored has been reset.")
            if knowledge:
                KnowledgeStorage().reset()
                click.echo("Knowledge has been reset.")

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while resetting the memories: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
