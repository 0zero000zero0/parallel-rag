import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from src.search_o1 import SearchO1
from src.utils import build_output_dir, read_jsonlines, write_jsonlines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search o1 runner")
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
    parser.add_argument("--is_chat", action="store_true",
                        help="Use /v1/chat/completions instead of /v1/completions")

    parser.add_argument("--max_search_limit", type=int, default=5)
    parser.add_argument("--docs_per_query", type=int, default=5)

    parser.add_argument("--search_max_tokens", type=int, default=512)
    parser.add_argument("--search_temperature", type=float, default=1.0)
    parser.add_argument("--search_top_p", type=float, default=0.9)

    parser.add_argument("--refine_max_tokens", type=int, default=1024)
    parser.add_argument("--refine_temperature", type=float, default=1.0)
    parser.add_argument("--refine_top_p", type=float, default=0.9)

    parser.add_argument("--stop_tokens", type=str, default=None)
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    input_path = Path(args.input_file)
    output_dir = build_output_dir(input_path,
                                  method_name="search-o1",
                                  model_name=args.model)

    output_path = output_dir / "output.jsonl"
    config_path = output_dir / "config.json"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    samples = read_jsonlines(input_path)
    if args.num_samples is not None:
        samples = samples[:max(0, args.num_samples)]

    pipeline = SearchO1.from_args(args)

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
        if args.debug:
            import debugpy
            debugpy.listen(9999)
            print("waiting for debugger attaching")
            debugpy.wait_for_client()
            print("debugger attached")
        try:
            batch_results = [
                dict(item)
                for item in pipeline.run_batch(questions, args.max_iterations)
            ]
        except Exception as exc:
            batch_error = str(exc)
            batch_results = [
                {
                    "query": question,
                    "search_steps": [],
                    "retrieved_docs": [],
                    "refine_outputs": [],
                    "final_answer": "",
                } for _, question, _ in validated_items
            ]

        for (idx, question, golden_answers), result in zip(validated_items,
                                                           batch_results):
            final_answer = result.get("final_answer", "")
            output_records.append({
                "index": idx,
                "question": question,
                "golden_answers": golden_answers,
                "model_output": final_answer,
                "predicted_answer_in_tag": final_answer,
                "pipeline_result": result,
                "error": batch_error,
            })

    write_jsonlines(output_path, output_records)


if __name__ == "__main__":
    main()
