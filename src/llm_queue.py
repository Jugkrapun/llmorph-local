"""
Shared Redis queue definitions for handing Hermes (the transformation LLM)
requests off to a separate worker process instead of calling the LLM
in-process.

`HermesQueueClient` (used by `llm_runner.py`) is the producer side: it
enqueues a request and waits for the matching reply. `llm_worker.py` is the
consumer side: it pops jobs off the same queue and does the actual
inference. Keeping both sides' queue/key naming here means they can never
drift apart.

All config lookups happen lazily (inside functions/methods, never at
import time) so this module behaves correctly regardless of whether it is
imported before or after the run config has been loaded.
"""
import json
import time
import sys
import uuid

import redis

from config_data import config_data

DEFAULT_QUEUE_NAME = "llmorph:hermes:requests"
RESPONSE_KEY_PREFIX = "llmorph:hermes:response:"
DEFAULT_RESPONSE_TIMEOUT = 300  # seconds a producer waits for one reply
RESPONSE_KEY_TTL = 3600  # seconds a reply is kept if never collected


def get_redis_client():
    return redis.Redis(
        host=config_data.get("redis_host") or "localhost",
        port=config_data.get("redis_port") or 6379,
        db=config_data.get("redis_db") or 0,
        decode_responses=True,
    )


def get_queue_name():
    return config_data.get("hermes_queue_name") or DEFAULT_QUEUE_NAME


def response_key(job_id):
    return f"{RESPONSE_KEY_PREFIX}{job_id}"


def make_job(model, messages):
    return {"job_id": str(uuid.uuid4()), "model": model, "messages": messages}


class HermesQueueClient:
    """
    Stands in for a direct 'calling object' (an OpenAI client bound to a
    model) that used to call Hermes synchronously in-process. Instead,
    `run()` enqueues the request on a Redis list and blocks only on
    collecting the reply from a separate `llm_worker.py` process, which
    owns the actual LLM call. This lets requests be queued and served
    asynchronously (by one or more workers) rather than each caller
    blocking the whole endpoint for the duration of inference.
    """

    def __init__(self, model, queue_name=None, response_timeout=None, wait_time=None, max_retries=None):
        self.model = model
        self.queue_name = queue_name or get_queue_name()
        self.response_timeout = response_timeout or config_data.get("hermes_response_timeout") or DEFAULT_RESPONSE_TIMEOUT
        self.wait_time = wait_time if wait_time is not None else (config_data.get("llm_wait_time") or 10)

        resolved_max_retries = max_retries if max_retries is not None else config_data.get("llm_max_retries")
        if resolved_max_retries is None or resolved_max_retries <= 0:
            resolved_max_retries = 9999999
        self.max_retries = resolved_max_retries

        self.redis = get_redis_client()

    def run(self, messages, max_retries=None):
        max_retries = max_retries or self.max_retries

        for attempt in range(max_retries):
            job = make_job(self.model, messages)
            r_key = response_key(job["job_id"])

            try:
                self.redis.rpush(self.queue_name, json.dumps(job))
                popped = self.redis.blpop([r_key], timeout=self.response_timeout)
            except redis.exceptions.RedisError as e:
                print(f"Redis error while submitting job to Hermes queue: {e}")
                popped = None

            if popped is None:
                print(f"Timed out waiting for a Hermes worker. Attempt {attempt + 1} of {max_retries}. Retrying...")
                time.sleep(self.wait_time)
                continue

            _, raw_result = popped
            result = json.loads(raw_result)

            if result.get("status") == "ok":
                return result["result"]

            # worker already exhausted its own inference retries for this job
            print(f"Hermes worker reported an error: {result.get('error')}")
            time.sleep(self.wait_time)

        print(f"Failed to get a response from the Hermes queue after {max_retries} attempts")
        sys.exit(1)
