import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from src.ircot import IRCOT
from src.utils import build_output_dir, read_jsonlines, write_jsonlines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IRCOT (Interleaving Retrieval with Chain-of-Thought) runner"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input jsonlines test file")
    parser.add_argument("--result_file", type=str, default=None,
                        help="Path to results jsonlines file")
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
    parser.add_argument("--model_path", type=str, default=None,
                        help="Local or HF model path used to load tokenizer for chat template")
    parser.add_argument("--llm_timeout", type=float, default=None)
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Render chat messages via tokenizer.apply_chat_template before /v1/completions")

    parser.add_argument("--generation_max_tokens", type=int, default=1024)
    parser.add_argument("--generation_temperature", type=float, default=0.7)
    parser.add_argument("--generation_top_p", type=float, default=0.9)

    parser.add_argument("--max_search_limit", type=int, default=5,
                        help="Maximum number of search queries per question")
    parser.add_argument("--max_iterations", type=int, default=10,
                        help="Maximum number of reasoning iterations")

    parser.add_argument("--stop_tokens", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    input_file_path = Path(args.input_file).expanduser()
    output_dir = build_output_dir(input_file_path,
                                  method_name="ircot",
                                  model_name=args.model)
    dataset_name = input_file_path.parent.name

    samples = read_jsonlines(input_file_path)
    if args.num_samples is not None:
        samples = samples[:max(0, args.num_samples)]

    pipeline = IRCOT.from_args(args)

    output_records: List[Dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))
    batch_starts = range(0, len(samples), batch_size)
    if args.debug:
        import debugpy
        debugpy.listen(9999)
        print("waiting for debugger attaching")
        debugpy.wait_for_client()
        print("debugger attached")
        debugpy.breakpoint()
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
        try:
            batch_results = [dict(item)
                             for item in pipeline.run_batch(questions)]
        except Exception as exc:
            batch_error = str(exc)
            batch_results = [
                {
                    "query": question,
                    "max_iterations": args.max_iterations,
                    "executed_iterations": 0,
                    "search_count": 0,
                    "prompts": [],
                    "raw_outputs": [],
                    "steps": [],
                    "final_answer": "",
                    "boxed_answer": "",
                    "timing": {},
                } for _, question, _ in validated_items
            ]

        for (idx, question, golden_answers), result in zip(validated_items,
                                                           batch_results):
            final_answer = str(result.get("final_answer", "") or "")
            boxed_answer = str(result.get("boxed_answer", "") or "")
            output_records.append({
                "index": idx,
                "question": question,
                "golden_answers": golden_answers,
                "model_output": result.get("raw_outputs", [""])[-1] if result.get("raw_outputs") else "",
                "predicted_answer_in_tag": boxed_answer or final_answer,
                "pipeline_result": result,
                "error": batch_error,
            })

    if args.result_file:
        output_file_path = Path(args.result_file).expanduser()
    else:
        output_file_path = output_dir / f"{dataset_name}.jsonl"
        config_file_path = output_dir / "config.json"
        with config_file_path.open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=0)
    write_jsonlines(output_file_path, output_records)


if __name__ == "__main__":
    main()
