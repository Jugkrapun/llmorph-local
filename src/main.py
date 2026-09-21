from mt_main import run_using_config
import argparse


def main():
    parser = argparse.ArgumentParser(prog="llmorph", prefix_chars="-", 
                                     description="LLMorph: A framework for testing LLMs with metamorphic relations.",
                                     epilog="Paper: https://valerio-terragni.github.io/assets/pdf/cho-icsme-2025.pdf"
                                    )
    parser.add_argument("-l", "--llm", required=True, type=str, help="The name of the LLM to test.")
    parser.add_argument("-t", "--task", required=True, type=str, help="The name of the NLP task to test on.")
    parser.add_argument("-m", "--mr", required=True, type=str, help="The id of the metamorphic relation to test using.")
    parser.add_argument("-i", "--input-data", required=True, type=str, help="The path to the JSON file containing the inputs. Structured as an array of data points.")
    parser.add_argument("-o", "--base-dir", required=True, type=str, help="The path to the directory where caches and outputs will be stored.")
    parser.add_argument("-r", "--replace-perc", required=False, type=float, metavar="PERCENT", nargs="?", default=0.1, help="The ratio value for generating follow up inputs (range: 0.0 - 1.0, default: 0.1).")
    parser.add_argument("-n", "--num-threads", required=False, type=int, metavar="N", default=1, help="Number of data points to process concurrently (SUT/Hermes calls in flight at once, default: 1).")
    parser.add_argument("--parallel-input-transformation", action="store_true", help="Also run input_transformation (follow-up input generation) fully in parallel across threads, instead of serializing it. Only safe for MRs whose input_transformation only calls an LLM (e.g. ITGPT/ITGPTSentence) -- unsafe for ones using a local model (spaCy, nlpaug, KeyBERT, ...).")
    parser.add_argument("-T", "--transformation-llm", required=False, type=str, metavar="MODEL", default=None, help="Model to use for Hermes (the transformation LLM), e.g. the name shown in the llama.cpp server UI. Defaults to the same model passed as 'llm' (the SUT) if not given.")


    args = parser.parse_args()

    config = {
        "tasks": {args.task: [args.mr]},
        "llm_list": [args.llm],
        "existing_source_inputs": args.input_data,
        "dir_base_default": args.base_dir,
        "replace_perc": args.replace_perc,
        "num_threads": args.num_threads,
        "parallel_input_transformation": args.parallel_input_transformation,
    }
    # transformation (Hermes) model defaults to the same model under test unless overridden with -t
    config["llm_for_transformation"] = args.transformation_llm or args.llm

    run_using_config(config)

if __name__ == "__main__":
    main()
