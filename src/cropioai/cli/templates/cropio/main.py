#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from {{folder_name}}.cropio import {{cropio_name}}

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# cropio locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the cropio.
    """
    inputs = {
        'topic': 'AI LLMs',
        'current_year': str(datetime.now().year)
    }
    
    try:
        {{cropio_name}}().cropio().takeoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the cropio: {e}")


def train():
    """
    Train the cropio for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        {{cropio_name}}().cropio().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the cropio: {e}")

def replay():
    """
    Replay the cropio execution from a specific task.
    """
    try:
        {{cropio_name}}().cropio().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the cropio: {e}")

def test():
    """
    Test the cropio execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        {{cropio_name}}().cropio().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the cropio: {e}")
