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

from openai import OpenAI
from requests.exceptions import Timeout

from config_handler import get_run_config_from_json, store_run_config
from llm_queue import get_redis_client, get_queue_name, response_key, RESPONSE_KEY_TTL

POLL_TIMEOUT = 5  # seconds between checks of the request queue; lets the worker notice shutdown promptly

_shutdown_requested = False


def _handle_shutdown_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


def build_run_llm(client, wait_time, max_retries):
    """
    Ported from llm_runner.get_llm_function: the model now travels with
    each queued job instead of being bound up front, since one worker can
    serve requests for any model.
    """
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
                try:
                    # Ignore if the error is a content filtering error
                    filter_error_msgs = [
                        "Invalid response object from API: \'{\"detail\":\"Error code: 400 - {\\\'error\\\': {\\\'message\\\': \\\\\"The response was filtered due to the prompt triggering Azure OpenAI\\\'s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766\\\\\", \\\'type\\\': None, \\\'param\\\': \\\'prompt\\\', \\\'code\\\': \\\'content_filter\\\', \\\'status\\\': 400}}\"}\' (HTTP response code was 500)",
                        "Error code: 500 - {\'detail\': \'Error code: 400 - {\\\'error\\\': {\\\'message\\\': \"The response was filtered due to the prompt triggering Azure OpenAI\\\'s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766\", \\\'type\\\': None, \\\'param\\\': \\\'prompt\\\', \\\'code\\\': \\\'content_filter\\\', \\\'status\\\': 400}}\'}",
                    ]
                    if e.message in filter_error_msgs or e.user_message in filter_error_msgs or e.code == 'content_filter' or str(e) in filter_error_msgs or "The response was filtered due to the prompt triggering Azure OpenAI" in str(e):
                        print("Warning: Content filtering error")
                        return "The response was filtered due to the prompt triggering Azure OpenAI\'s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766"
                    rep_error_msgs = [
                        "An error occurred: Error code: 500 - {\'detail\': \'Error code: 400 - {\\\'error\\\': {\\\'message\\\': \"Sorry! We\\\'ve encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt.\", \\\'type\\\': \\\'invalid_request_error\\\', \\\'param\\\': \\\'prompt\\\', \\\'code\\\': \\\'invalid_prompt\\\'}}\'}"
                    ]
                    if e.message in rep_error_msgs or e.user_message in rep_error_msgs or str(e) in rep_error_msgs:
                        print("Warning: Repetitive patterns error")
                        return "Sorry! We\'ve encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt."
                except:
                    pass

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
        print(f"Skipping job with no job_id: {job}")
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


def worker_loop(run_llm):
    """
    One polling loop, meant to run on its own thread: each thread gets its
    own Redis client (a blocking BLPOP call ties up whatever connection it
    borrows for up to POLL_TIMEOUT seconds, so sharing one client across
    threads would serialize them right back). Running N of these threads in
    one process lets that single process hold N jobs in flight at once,
    instead of needing N separate `python src/llm_worker.py` processes.
    """
    r = get_redis_client()
    queue_name = get_queue_name()
    while not _shutdown_requested:
        popped = r.blpop([queue_name], timeout=POLL_TIMEOUT)
        if popped is None:
            continue
        _, raw_job = popped
        process_job(raw_job, run_llm, r)


def main():
    parser = argparse.ArgumentParser(description="Hermes queue worker: pops jobs off the Redis queue and runs inference for them.")
    parser.add_argument("-n", "--concurrency", type=int, default=1, metavar="N", help="Number of jobs this single process handles concurrently (default: 1). Raise this instead of running multiple `python src/llm_worker.py` processes.")
    args = parser.parse_args()

    run_config = get_run_config_from_json()
    store_run_config(run_config)  # keeps config_data (used by llm_queue helpers) in sync with this process

    endpoint = run_config.get("llm_endpoint")
    wait_time = run_config.get("llm_wait_time") or 10
    max_retries = run_config.get("llm_max_retries")
    if max_retries is None or max_retries <= 0:
        max_retries = 9999999

    client = OpenAI(api_key="foot", base_url=endpoint)
    run_llm = build_run_llm(client, wait_time, max_retries)
    queue_name = get_queue_name()

    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    print(f"Hermes queue worker started ({args.concurrency} concurrent slot(s)). Listening on '{queue_name}' -> {endpoint}. Press Ctrl+C to stop.")

    threads = [threading.Thread(target=worker_loop, args=(run_llm,), daemon=True) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("Shutting down Hermes queue worker.")


if __name__ == "__main__":
    main()
