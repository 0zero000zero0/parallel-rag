import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.search_o1 import SearchO1
from src.utils import build_output_dir, percentile, read_jsonlines, write_jsonlines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search o1 runner")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input jsonlines test file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for batched testing"
    )

    parser.add_argument(
        "--retriever_base_url", type=str, default="http://127.0.0.1:9100"
    )
    parser.add_argument("--retriever_top_k", type=int, default=5)
    parser.add_argument("--retriever_timeout", type=float, default=None)

    parser.add_argument("--openai_base_url", type=str, default="http://127.0.0.1:8001")
    parser.add_argument("--openai_api_key", type=str, default="TEST")
    parser.add_argument("--model", type=str, default="Qwen3-14B")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Local or HF model path used to load tokenizer for chat template",
    )
    parser.add_argument("--llm_timeout", type=float, default=None)
    parser.add_argument(
        "--is_chat",
        action="store_true",
        help="Use /v1/chat/completions instead of /v1/completions",
    )
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Render chat messages via tokenizer.apply_chat_template before /v1/completions",
    )

    parser.add_argument("--max_search_limit", type=int, default=5)

    parser.add_argument("--search_max_tokens", type=int, default=512)
    parser.add_argument("--search_temperature", type=float, default=1.0)
    parser.add_argument("--search_top_p", type=float, default=0.9)

    parser.add_argument("--refine_max_tokens", type=int, default=1024)
    parser.add_argument("--refine_temperature", type=float, default=1.0)
    parser.add_argument("--refine_top_p", type=float, default=0.9)

    parser.add_argument("--stop_tokens", type=str, default=None)
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--result_file", type=str, default=None, help="Path to results jsonlines file"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_top_dir", default="outputs", type=str)
    return parser


def main() -> None:
    run_started_ns = time.perf_counter_ns()
    parser = build_parser()
    args = parser.parse_args()
    input_path = Path(args.input_file).expanduser()
    output_dir = build_output_dir(
        input_path,
        method_name="search-o1",
        model_name=args.model,
        top_dir=args.output_top_dir,
    )
    dataset_name = input_path.parent.name

    read_started_ns = time.perf_counter_ns()
    samples = read_jsonlines(input_path)
    read_data_ms = (time.perf_counter_ns() - read_started_ns) / 1_000_000.0
    if args.num_samples is not None:
        samples = samples[: max(0, args.num_samples)]

    pipeline = SearchO1.from_args(args)

    output_records: list[dict[str, Any]] = []
    batch_timings: list[dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))
    batch_starts = range(0, len(samples), batch_size)

    if args.debug:
        import debugpy

        debugpy.listen(9999)
        print("waiting for debugger attaching")
        debugpy.wait_for_client()
        print("debugger attached")

    pipeline_started_ns = time.perf_counter_ns()
    for batch_idx, batch_start in enumerate(
        tqdm(batch_starts, desc="Processing batches", unit="batch")
    ):
        batch_samples = samples[batch_start : batch_start + batch_size]
        questions: list[str] = []
        validated_items: list[tuple[int, str, list[str]]] = []
        for offset, sample in enumerate(batch_samples):
            idx = batch_start + offset
            question = sample.get("question", "")
            golden_answers = sample.get("golden_answers", [])
            validated_items.append((idx, question, golden_answers))
            questions.append(question)

        batch_results: list[dict[str, Any]] = []
        batch_error = ""
        try:
            batch_started_ns = time.perf_counter_ns()
            batch_results = [
                dict(item)
                for item in pipeline.run_batch(questions, args.max_iterations)
            ]
            batch_wall_ms = (time.perf_counter_ns() - batch_started_ns) / 1_000_000.0
        except Exception as exc:
            batch_error = str(exc)
            batch_wall_ms = 0.0
            batch_results = [
                {
                    "query": question,
                    "search_steps": [],
                    "retrieved_docs": [],
                    "refine_outputs": [],
                    "final_answer": "",
                    "timing": {},
                }
                for _, question, _ in validated_items
            ]

        batch_timing = dict(getattr(pipeline, "latest_batch_timing", {}))
        batch_timing.update(
            {
                "batch_index": batch_idx,
                "batch_start": batch_start,
                "batch_size": len(batch_samples),
                "batch_wall_ms": batch_wall_ms,
                "error": batch_error,
            }
        )
        batch_timings.append(batch_timing)

        for (idx, question, golden_answers), result in zip(
            validated_items, batch_results
        ):
            final_answer = result.get("final_answer", "")
            output_records.append(
                {
                    "index": idx,
                    "question": question,
                    "golden_answers": golden_answers,
                    "model_output": final_answer,
                    "predicted_answer_in_tag": final_answer,
                    "timing": result.get("timing", {}),
                    "pipeline_result": result,
                    "error": batch_error,
                }
            )

    pipeline_total_ms = (time.perf_counter_ns() - pipeline_started_ns) / 1_000_000.0

    if args.result_file:
        output_path = Path(args.result_file).expanduser()
    else:
        output_path = output_dir / f"{dataset_name}.jsonl"
        config_path = output_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=0)

    write_started_ns = time.perf_counter_ns()
    write_jsonlines(output_path, output_records)
    write_result_ms = (time.perf_counter_ns() - write_started_ns) / 1_000_000.0

    sample_total_ms = [
        float(record.get("timing", {}).get("total_ms", 0.0))
        for record in output_records
    ]
    batch_wall_ms_values = [
        float(item.get("batch_wall_ms", 0.0)) for item in batch_timings
    ]

    timing_summary: dict[str, Any] = {
        "sample_count": len(samples),
        "batch_count": len(batch_timings),
        "read_data_ms": read_data_ms,
        "pipeline_total_ms": pipeline_total_ms,
        "write_result_ms": write_result_ms,
        "total_run_ms": (time.perf_counter_ns() - run_started_ns) / 1_000_000.0,
        "sample_timing_stats_ms": {
            "avg": sum(sample_total_ms) / len(sample_total_ms)
            if sample_total_ms
            else 0.0,
            "p50": percentile(sample_total_ms, 0.5),
            "p95": percentile(sample_total_ms, 0.95),
            "max": max(sample_total_ms) if sample_total_ms else 0.0,
        },
        "batch_wall_stats_ms": {
            "avg": sum(batch_wall_ms_values) / len(batch_wall_ms_values)
            if batch_wall_ms_values
            else 0.0,
            "p50": percentile(batch_wall_ms_values, 0.5),
            "p95": percentile(batch_wall_ms_values, 0.95),
            "max": max(batch_wall_ms_values) if batch_wall_ms_values else 0.0,
        },
        "batch_timings": batch_timings,
    }

    timing_output_path = output_dir / "time.json"
    with timing_output_path.open("w", encoding="utf-8") as f:
        json.dump(timing_summary, f, ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
