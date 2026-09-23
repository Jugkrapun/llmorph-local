"""
Queue receiver for Hermes (the transformation LLM).

This is the only part of the system that actually calls the Hermes LLM
endpoint for transformation requests: LLMorph itself (llm_runner.py) only
ever enqueues a request onto a Redis list and waits for the matching
reply. That split lets many transformation requests be queued up and
served asynchronously, by one or more of these worker processes, instead
of each request blocking LLMorph's own process for the duration of
inference.

Run one (or several, for more throughput) with, from the repository root:
    python src/llm_worker.py

Stop with Ctrl+C.
"""
import argparse
import json
import signal
import threading
import time
import os

from openai import OpenAI
from requests.exceptions import Timeout

from config_handler import get_run_config_from_json, store_run_config
from llm_queue import get_redis_client, get_queue_name, response_key, RESPONSE_KEY_TTL

POLL_TIMEOUT = 5

_shutdown_requested = False

def _handle_shutdown_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True

def build_run_llm(client, wait_time, max_retries):
    def run_llm(model, messages):
        for attempt in range(max_retries):
            try:
                chat_completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                )
                return str(chat_completion.choices[0].message.content)

            except Timeout:
                print(f"Request timed out. Attempt {attempt + 1} of {max_retries}. Retrying...")
            except Exception as e:
                # Fix: Refactored fragile error handling. Removed bare except.
                error_str = str(e).lower()
                
                if "content management policy" in error_str or "content_filter" in error_str:
                    print("Warning: Content filtering error")
                    return "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry."
                
                if "repetitive patterns" in error_str:
                    print("Warning: Repetitive patterns error")
                    return "Sorry! We've encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt."

                print(f"An error occurred: {e}")
                print(f"Attempt {attempt + 1} of {max_retries}. Retrying...")
                time.sleep(wait_time)

        print(f"Failed to get response after {max_retries} attempts")
        return None

    return run_llm

def process_job(raw_job, run_llm, r):
    try:
        job = json.loads(raw_job)
    except json.JSONDecodeError:
        print(f"Skipping malformed job payload: {raw_job}")
        return

    job_id = job.get("job_id")
    model = job.get("model")
    messages = job.get("messages")

    if not job_id:
        return

    print(f"Running inference for job {job_id} (model={model})...")
    generated_text = run_llm(model, messages)

    if generated_text is None:
        payload = {"status": "error", "error": "Failed to get a response from the LLM after all retries"}
    else:
        payload = {"status": "ok", "result": generated_text}

    key = response_key(job_id)
    r.rpush(key, json.dumps(payload))
    r.expire(key, RESPONSE_KEY_TTL)

def worker_loop(run_llm, queue_name):
    r = get_redis_client()
    while not _shutdown_requested:
        popped = r.blpop([queue_name], timeout=POLL_TIMEOUT)
        if popped is None:
            continue
        _, raw_job = popped
        process_job(raw_job, run_llm, r)

def main():
    parser = argparse.ArgumentParser(description="LLM queue worker: pops jobs off the Redis queue and runs inference.")
    parser.add_argument("-n", "--concurrency", type=int, default=1, metavar="N", help="Number of jobs to handle concurrently.")
    
    # Require the model name to listen to the correct queue
    parser.add_argument("-m", "--model", type=str, required=True, help="Target model name to listen for in the queue.")
    
    # Allow overriding the endpoint specifically for this worker instance
    parser.add_argument("-e", "--endpoint", type=str, default=None, help="Specific LLM endpoint URL. Overrides config data if provided.")
    args = parser.parse_args()

    run_config = get_run_config_from_json()
    store_run_config(run_config)

    endpoint = args.endpoint or run_config.get("llm_endpoint")
    wait_time = run_config.get("llm_wait_time") or 10
    max_retries = run_config.get("llm_max_retries")
    
    # Fix: Limit infinite retries
    if max_retries is None or max_retries <= 0:
        max_retries = 5

    # Fix: Read API key securely from environment variable, default to dummy string if not set
    api_key = os.environ.get("OPENAI_API_KEY", "sk-dummy")
    client = OpenAI(api_key=api_key, base_url=endpoint)
    
    run_llm = build_run_llm(client, wait_time, max_retries)
    queue_name = get_queue_name(args.model)

    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    print(f"LLM Queue Worker for '{args.model}' started ({args.concurrency} concurrent slot(s)).")
    print(f"Listening on '{queue_name}' -> {endpoint}. Press Ctrl+C to stop.")

    threads = [threading.Thread(target=worker_loop, args=(run_llm, queue_name), daemon=True) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("Shutting down LLM queue worker.")

if __name__ == "__main__":
    main()
