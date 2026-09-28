from typing import List, Callable, Optional
from llm_handler import run_template_llm
from llm_queue import LLMQueueClient

from config_data import config_data

llm_for_transformation = config_data.get("llm_for_transformation")

def get_llm_function(model: str) -> Callable:
    """Returns a callable function bound to a specific model via the Redis queue."""
    queue_client = LLMQueueClient(model=model)

    def run_llm(messages: list, max_retries: int = None) -> Optional[str]:
        return queue_client.run(messages, max_retries=max_retries)

    return run_llm

# Shared client for input transformation tasks
_transformation_queue_client = LLMQueueClient(model=llm_for_transformation)

def run_template_gpt(inputs: list, prompt_template: str, examples: list = [], placeholder_template: str = "{INPUT_#}") -> Optional[str]:
    if not isinstance(inputs, list):
        inputs = [inputs]
    return run_template_llm(_transformation_queue_client.run, inputs, prompt_template, examples, placeholder_template)
