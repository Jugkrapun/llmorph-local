from llm_handler import run_template_llm
from llm_queue import HermesQueueClient

from config_data import config_data

llm_for_transformation = config_data.get("llm_for_transformation")

# Both the SUT (model under test) and Hermes (the transformation LLM) are
# called through the same Redis-backed queue (see llm_queue.py). A job
# carries its own model name, so llm_worker.py can serve requests for any
# model -- running more llm_worker.py processes is what makes concurrent
# SUT calls (and concurrent Hermes calls) actually run in parallel, not
# just get submitted in parallel.

def get_llm_function(model):
    queue_client = HermesQueueClient(model=model)

    def run_llm(messages, max_retries=None):
        return queue_client.run(messages, max_retries=max_retries)

    return run_llm


# this queue client is used as a tool; this llm is not being tested
_hermes_queue_client = HermesQueueClient(model=llm_for_transformation)

def run_template_gpt(inputs : list, prompt_template : str, examples : list=[], placeholder_template="{INPUT_#}") -> str | None:
    if not isinstance(inputs, list):
        inputs = [inputs]
    return run_template_llm(_hermes_queue_client.run, inputs, prompt_template, examples, placeholder_template)
