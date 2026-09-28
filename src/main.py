from mt_main import run_using_config
import argparse

def main():
    parser = argparse.ArgumentParser(description="LLMorph: A framework for testing LLMs with metamorphic relations.")
    parser.add_argument("llm", type=str, help="The name of the LLM to test.")
    parser.add_argument("task", type=str, help="The name of the NLP task to test on.")
    parser.add_argument("mr", type=str, help="The name of the metamorphic relation to test using.")
    parser.add_argument("input_data", type=str, help="The path to the JSON file containing the inputs.")
    parser.add_argument("base_dir", type=str, help="The path to the directory where caches and outputs will be stored.")
    
    ########################################################################
    parser.add_argument("-r", "--replace-perc", required=False, type=float, metavar="PERCENT", nargs="?", default=0.1, help="The ratio value for generating follow up inputs (range: 0.0 - 1.0, default: 0.1).")
    
    # Set sensible defaults for parallel execution directly in argparse
    parser.add_argument("-n", "--num-threads", required=False, type=int, metavar="N", default=10, help="Number of data points to process concurrently (default: 10).")
    parser.add_argument("--parallel-input-transformation", action="store_true", default=True, help="Run input transformation in parallel (default: True).")
    parser.add_argument("-t", "--transformation-llm", required=False, type=str, metavar="MODEL", default="Nous-Hermes-2-Mixtral-8x7B-DPO-Q2_K", help="Model to use for the transformation LLM. Defaults to the SUT model.")

    args = parser.parse_args()

    config = {
        "tasks": {args.task: [args.mr]},
        "llm_list": [args.llm],
        "existing_source_inputs": args.input_data,
        "dir_base_default": args.base_dir,
        "replace_perc": args.replace_perc,
        
        # Values naturally cascade from argparse defaults
        "num_threads": args.num_threads,
        "parallel_input_transformation": args.parallel_input_transformation,
    }
    
    # Transformation model defaults to the same model under test if not overridden
    config["llm_for_transformation"] = args.transformation_llm or args.llm

    run_using_config(config)

if __name__ == "__main__":
    main()
