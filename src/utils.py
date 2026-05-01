import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonlines(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            records.append(obj)
    return records


def write_jsonlines(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_output_dir(
    input_path: Path, method_name: str, model_name: str, top_dir: str = "outputs"
) -> Path:
    dataset_name = input_path.parent.name
    dataset_root = Path(top_dir) / method_name / model_name / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)

    existing_indices: List[int] = []
    for child in dataset_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            existing_indices.append(int(child.name))

    next_index = 0 if not existing_indices else max(existing_indices) + 1
    output_dir = dataset_root / str(next_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_tensorboard_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(log_dir))
    except Exception:
        try:
            from tensorboardX import SummaryWriter

            return SummaryWriter(log_dir=str(log_dir))
        except Exception:
            return None


def percentile(values: List[float], p: float) -> float:
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
