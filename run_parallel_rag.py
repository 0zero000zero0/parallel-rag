import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from src.parallel_refiner import ParallelRetrieverRefiner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel retrieve-refine-synthesize runner")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input jsonlines test file")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for batched testing")

    parser.add_argument("--retriever_base_url", type=str,
                        default="http://127.0.0.1:9100")
    parser.add_argument("--retriever_top_k", type=int, default=5)
    parser.add_argument("--retriever_timeout", type=float, default=None)

    parser.add_argument("--openai_base_url", type=str,
                        default="http://127.0.0.1:8001")
    parser.add_argument("--openai_api_key", type=str, default="TEST")
    parser.add_argument("--model", type=str, default="Qwen3-14B")
    parser.add_argument("--llm_timeout", type=float, default=None)

    parser.add_argument("--parallel_query_count", type=int, default=3)
    parser.add_argument("--docs_per_query", type=int, default=5)

    parser.add_argument("--query_max_tokens", type=int, default=1024)
    parser.add_argument("--query_temperature", type=float, default=1.0)
    parser.add_argument("--query_top_p", type=float, default=0.9)

    parser.add_argument("--refine_max_tokens", type=int, default=1024)
    parser.add_argument("--refine_temperature", type=float, default=0.6)
    parser.add_argument("--refine_top_p", type=float, default=0.9)

    parser.add_argument("--synthesize_max_tokens", type=int, default=1024)
    parser.add_argument("--synthesize_temperature", type=float, default=0.6)
    parser.add_argument("--synthesize_top_p", type=float, default=0.9)

    parser.add_argument("--stop_tokens", type=str,
                        default=None)
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--debug",  action='store_true')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    utils = importlib.import_module("src.utils")

    input_path = Path(args.input_file)
    output_dir = utils.build_output_dir(input_path,
                                        method_name="parallel_rag",
                                        model_name=args.model)

    output_path = output_dir / "output.jsonl"
    config_path = output_dir / "config.json"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    samples = utils.read_jsonlines(input_path)
    if args.num_samples is not None:
        samples = samples[:max(0, args.num_samples)]
    refiner = ParallelRetrieverRefiner.from_args(args)

    output_records: List[Dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))
    batch_starts = range(0, len(samples), batch_size)
    for batch_start in tqdm(batch_starts,
                            desc="Processing batches",
                            unit="batch"):
        batch_samples = samples[batch_start:batch_start + batch_size]
        questions: List[str] = []
        validated_items: List[tuple[int, str, List[str]]] = []
        for offset, sample in enumerate(batch_samples):
            idx = batch_start + offset
            question = sample.get("question", "")
            golden_answers = sample.get("golden_answers", [])
            validated_items.append((idx, question, golden_answers))
            questions.append(question)

        batch_results: List[Dict[str, Any]] = []
        batch_error = ""
        # if args.debug is True:
        #     import debugpy
        #     debugpy.listen(9999)
        #     print(f"waiting for debugger attaching")
        #     debugpy.wait_for_client()
        #     print(f"debugger attached")
        try:
            batch_results = [dict(item)
                             for item in refiner.run_batch(questions, args.max_iterations)]
        except Exception as exc:
            batch_error = str(exc)
            batch_results = [
                {
                    "query": question,
                    "parallel_queries": [],
                    "retrieved_docs": [],
                    "refined_contexts": [],
                    "final_answer": "",
                } for _, question, _ in validated_items
            ]

        for (idx, question, golden_answers), result in zip(validated_items,
                                                           batch_results):
            final_answer = result.get("final_answer", "")
            answer_in_tag = final_answer
            output_records.append({
                "index": idx,
                "question": question,
                "golden_answers": golden_answers,
                "model_output": final_answer,
                "predicted_answer_in_tag": answer_in_tag,
                "pipeline_result": result,
                "error": batch_error,
            })

    utils.write_jsonlines(output_path, output_records)


if __name__ == "__main__":
    main()
