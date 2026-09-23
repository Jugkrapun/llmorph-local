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

DEFAULT_RESPONSE_TIMEOUT = 300
RESPONSE_KEY_TTL = 3600

def get_redis_client():
    return redis.Redis(
        host=config_data.get("redis_host") or "localhost",
        port=config_data.get("redis_port") or 6379,
        db=config_data.get("redis_db") or 0,
        decode_responses=True,
    )

def get_queue_name(model):
    # Fix: Isolate queues by model name to support parallel execution for multiple models
    return f"llmorph:queue:{model}"

def response_key(job_id):
    return f"llmorph:response:{job_id}"

def make_job(model, messages):
    return {"job_id": str(uuid.uuid4()), "model": model, "messages": messages}

class LLMQueueClient:
    def __init__(self, model, response_timeout=None, wait_time=None, max_retries=None):
        self.model = model
        self.queue_name = get_queue_name(model)
        self.response_timeout = response_timeout or config_data.get("llm_response_timeout") or DEFAULT_RESPONSE_TIMEOUT
        self.wait_time = wait_time if wait_time is not None else (config_data.get("llm_wait_time") or 10)

        resolved_max_retries = max_retries if max_retries is not None else config_data.get("llm_max_retries")
        # Fix: Change fallback from near-infinite 9999999 to a sensible default (5)
        if resolved_max_retries is None or resolved_max_retries <= 0:
            resolved_max_retries = 5 
        self.max_retries = resolved_max_retries
        self.redis = get_redis_client()

    def run(self, messages, max_retries=None):
        max_retries = max_retries or self.max_retries

        for attempt in range(max_retries):
            job = make_job(self.model, messages)
            r_key = response_key(job["job_id"])
            job_json = json.dumps(job)

            try:
                self.redis.rpush(self.queue_name, job_json)
                popped = self.redis.blpop([r_key], timeout=self.response_timeout)
            except redis.exceptions.RedisError as e:
                print(f"Redis error while submitting job to {self.model} queue: {e}")
                popped = None

            if popped is None:
                print(f"Timed out waiting for {self.model} worker. Attempt {attempt + 1} of {max_retries}. Retrying...")
                # Fix: Resource leak solved by removing the timed-out job from the queue
                self.redis.lrem(self.queue_name, 0, job_json)
                time.sleep(self.wait_time)
                continue

            _, raw_result = popped
            result = json.loads(raw_result)

            if result.get("status") == "ok":
                return result["result"]

            print(f"Worker for {self.model} reported an error: {result.get('error')}")
            time.sleep(self.wait_time)

        print(f"Failed to get a response from the {self.model} queue after {max_retries} attempts")
        sys.exit(1)
