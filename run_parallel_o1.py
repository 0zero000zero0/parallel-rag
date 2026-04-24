import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from src.parallel_o1 import ParallelO1
from src.utils import build_output_dir, read_jsonlines, write_jsonlines


def _create_tensorboard_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=str(log_dir))
    except Exception:
        try:
            from tensorboardX import SummaryWriter
            return SummaryWriter(log_dir=str(log_dir))
        except Exception:
            return None


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel o1 runner")
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

    parser.add_argument("--docs_per_query", type=int, default=5)

    parser.add_argument("--navigator_agent_max_tokens", type=int, default=256)
    parser.add_argument("--navigator_agent_temperature",
                        type=float, default=0.6)
    parser.add_argument("--navigator_agent_top_p", type=float, default=0.9)

    parser.add_argument("--path_agent_max_tokens", type=int, default=512)
    parser.add_argument("--path_agent_temperature", type=float, default=0.8)
    parser.add_argument("--path_agent_top_p", type=float, default=0.9)

    parser.add_argument("--refine_max_tokens", type=int, default=1024)
    parser.add_argument("--refine_temperature", type=float, default=0.8)
    parser.add_argument("--refine_top_p", type=float, default=0.9)

    parser.add_argument("--stop_tokens", type=str,
                        default="</search>,</answer>,</search_directions>")
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--timing_json_file", type=str,
                        default="timing_parallel_o1.json",
                        help="File name under output_dir to persist timing metrics json")
    parser.add_argument("--debug",  action='store_true')
    return parser


def main() -> None:
    run_started_ns = time.perf_counter_ns()
    parser = build_parser()
    args = parser.parse_args()
    input_file_path = Path(args.input_file)
    output_dir = build_output_dir(input_file_path,
                                  method_name="parallel-o1",
                                  model_name=args.model)
    dataset_name = input_file_path.parent.name

    read_started_ns = time.perf_counter_ns()
    samples = read_jsonlines(input_file_path)
    read_data_ms = (time.perf_counter_ns() - read_started_ns) / 1_000_000.0
    if args.num_samples is not None:
        samples = samples[:max(0, args.num_samples)]

    pipeline = ParallelO1.from_args(args)

    output_records: List[Dict[str, Any]] = []
    batch_timings: List[Dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))
    batch_starts = range(0, len(samples), batch_size)

    writer = _create_tensorboard_writer(output_dir / "tensorboard")

    if args.debug:
        import debugpy
        debugpy.listen(9999)
        print("waiting for debugger attaching")
        debugpy.wait_for_client()
        print("debugger attached")
        print(f"break point")
        debugpy.breakpoint()
    pipeline_started_ns = time.perf_counter_ns()
    for batch_idx, batch_start in enumerate(tqdm(batch_starts,
                                                 desc="Processing batches",
                                                 unit="batch")):
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

        batch_started_ns = time.perf_counter_ns()
        batch_results = [dict(item)
                         for item in pipeline.run_batch(questions, args.max_iterations)]
        batch_wall_ms = (time.perf_counter_ns() -
                         batch_started_ns) / 1_000_000.0
        batch_timing = dict(pipeline.latest_batch_timing)
        batch_timing.update({
            "batch_index": batch_idx,
            "batch_start": batch_start,
            "batch_size": len(batch_samples),
            "batch_wall_ms": batch_wall_ms,
        })
        batch_timings.append(batch_timing)

        if writer is not None:
            writer.add_scalar("batch/batch_wall_ms", batch_wall_ms, batch_idx)
            phase_totals = batch_timing.get("phase_totals_ms", {})
            for phase_name, phase_ms in phase_totals.items():
                writer.add_scalar(f"batch/{phase_name}",
                                  float(phase_ms),
                                  batch_idx)

        for (idx, question, golden_answers), result in zip(validated_items,
                                                           batch_results):
            final_answer = result.get("final_answer", "")
            output_records.append({
                "index": idx,
                "question": question,
                "golden_answers": golden_answers,
                "predicted_answer_in_tag": final_answer,
                "timing": result.get("timing", {}),
                "pipeline_result": result,
            })

    pipeline_total_ms = (time.perf_counter_ns() -
                         pipeline_started_ns) / 1_000_000.0

    if args.result_file:
        output_file_path = Path(args.result_file)
    else:
        output_file_path = output_dir / f"{dataset_name}.jsonl"
        config_file_path = output_dir / "config.json"
        with config_file_path.open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=0)

    write_started_ns = time.perf_counter_ns()
    write_jsonlines(output_file_path, output_records)
    write_result_ms = (time.perf_counter_ns() - write_started_ns) / 1_000_000.0

    sample_total_ms = [
        float(record.get("timing", {}).get("total_ms", 0.0))
        for record in output_records
    ]
    batch_wall_ms_values = [float(item.get("batch_wall_ms", 0.0))
                            for item in batch_timings]

    timing_summary: Dict[str, Any] = {
        "sample_count": len(samples),
        "batch_count": len(batch_timings),
        "read_data_ms": read_data_ms,
        "pipeline_total_ms": pipeline_total_ms,
        "write_result_ms": write_result_ms,
        "total_run_ms": (time.perf_counter_ns() - run_started_ns) / 1_000_000.0,
        "sample_timing_stats_ms": {
            "avg": sum(sample_total_ms) / len(sample_total_ms) if sample_total_ms else 0.0,
            "p50": _percentile(sample_total_ms, 0.5),
            "p95": _percentile(sample_total_ms, 0.95),
            "max": max(sample_total_ms) if sample_total_ms else 0.0,
        },
        "batch_wall_stats_ms": {
            "avg": sum(batch_wall_ms_values) / len(batch_wall_ms_values) if batch_wall_ms_values else 0.0,
            "p50": _percentile(batch_wall_ms_values, 0.5),
            "p95": _percentile(batch_wall_ms_values, 0.95),
            "max": max(batch_wall_ms_values) if batch_wall_ms_values else 0.0,
        },
        "batch_timings": batch_timings,
    }

    if writer is not None:
        writer.add_scalar("run/read_data_ms",
                          timing_summary["read_data_ms"], 0)
        writer.add_scalar("run/pipeline_total_ms",
                          timing_summary["pipeline_total_ms"],
                          0)
        writer.add_scalar("run/write_result_ms",
                          timing_summary["write_result_ms"],
                          0)
        writer.add_scalar("run/total_run_ms",
                          timing_summary["total_run_ms"], 0)
        writer.add_scalar("sample/total_ms_avg",
                          timing_summary["sample_timing_stats_ms"]["avg"],
                          0)
        writer.add_scalar("sample/total_ms_p95",
                          timing_summary["sample_timing_stats_ms"]["p95"],
                          0)
        writer.flush()
        writer.close()

    timing_output_path = output_dir / "time.json"
    with timing_output_path.open("w", encoding="utf-8") as f:
        json.dump(timing_summary, f, ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
