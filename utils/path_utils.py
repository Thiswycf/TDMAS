import os
import re
from typing import Iterable, Literal, Optional, Tuple

TaskType = Literal["optimize", "inference"]

WORKSPACE_ROOT = "tdmas_workspace"


def workspace_path(*parts: str) -> str:
    """Join parts under the workspace root."""
    return os.path.join(WORKSPACE_ROOT, *[str(p) for p in parts])


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def ensure_parent_dir(file_path: str) -> None:
    """Create parent directory for a file path."""
    parent = os.path.dirname(file_path)
    if parent:
        ensure_dir(parent)


def resolve_existing_path(preferred: str, legacy_candidates: Iterable[str] = ()) -> str:
    """Return the first existing path among preferred and legacy candidates."""
    for candidate in (preferred, *legacy_candidates):
        if candidate and os.path.exists(candidate):
            return candidate
    return preferred


def workflow_output_dir(data_set: str, zcp: str) -> str:
    return workspace_path("output_workflow", data_set, zcp)


def workflow_output_file(data_set: str, zcp: str, epoch: int | str, task_type: TaskType) -> str:
    suffix = "-test" if task_type == "inference" else ""
    return os.path.join(workflow_output_dir(data_set, zcp), f"dataset-{epoch}{suffix}.pkl")


def legacy_workflow_output_file(zcp: str, epoch: int | str, task_type: TaskType) -> str:
    suffix = "-test" if task_type == "inference" else ""
    return workspace_path("output_workflow", zcp, f"dataset-{epoch}{suffix}.pkl")


def evaluation_output_dir(data_set: str, zcp: str) -> str:
    return workspace_path("output_evaluation", data_set, zcp)


def evaluation_output_file(
    data_set: str, zcp: str, epoch: int | str, parallel_id: int | str, task_type: TaskType
) -> str:
    suffix = "-test" if task_type == "inference" else ""
    return os.path.join(evaluation_output_dir(data_set, zcp), f"scores-{epoch}-{parallel_id}{suffix}.txt")


def legacy_evaluation_output_file(zcp: str, epoch: int | str, parallel_id: int | str, task_type: TaskType) -> str:
    suffix = "-test" if task_type == "inference" else ""
    return workspace_path("output_evaluation", zcp, f"scores-{epoch}-{parallel_id}{suffix}.txt")


def solve_rate_file(data_set: str, zcp: str, task_type: TaskType) -> str:
    suffix = "_test" if task_type == "inference" else "_train"
    return os.path.join(
        evaluation_output_dir(data_set, zcp),
        f"solve_rate{suffix}_{data_set}.txt",
    )


def temp_eval_dir(data_set: str, zcp: str) -> str:
    return workspace_path("temp_eval_workflow_file", data_set, zcp)


def temp_generate_dir(data_set: str, zcp: str) -> str:
    return workspace_path("temp_gene_workflow_file", data_set, zcp)

def wsft_data_dir(data_set: str, zcp: str) -> str:
    return workspace_path("output_wsft_data", data_set, zcp)


def wsft_data_file(data_set: str, zcp: str, epoch: int | str) -> str:
    return os.path.join(wsft_data_dir(data_set, zcp), f"wsft_data-{epoch}.pkl")


def preference_data_dir(data_set: str, zcp: str) -> str:
    return workspace_path("output_preference_data", data_set, zcp)


def preference_data_file(data_set: str, zcp: str, epoch: int | str) -> str:
    return os.path.join(preference_data_dir(data_set, zcp), f"preference_data-{epoch}.pkl")


def legacy_preference_data_file(zcp: str, epoch: int | str) -> str:
    return workspace_path("output_preference_data", zcp, f"preference_data-{epoch}.pkl")


def finetuned_dir(data_set: str, zcp: str) -> str:
    return workspace_path("finetuned", data_set, zcp)


def finetuned_epoch_dir(data_set: str, zcp: str, epoch: int | str) -> str:
    return os.path.join(finetuned_dir(data_set, zcp), str(epoch))


def finetuned_model_path(data_set: str, zcp: str, epoch: int | str) -> str:
    # return os.path.join(finetuned_epoch_dir(data_set, zcp, epoch), "merged")
    return os.path.join(finetuned_epoch_dir(data_set, zcp, epoch), "final")


def legacy_finetuned_model_path(zcp: str, epoch: int | str) -> str:
    # return workspace_path("finetuned", zcp, str(epoch), "merged")
    return workspace_path("finetuned", zcp, str(epoch), "final")


_FINETUNED_NAME_PATTERN = re.compile(
    r"_finetuned(?:_DATASET(?P<data_set>.+?))?_ZCP(?P<zcp>.+?)_EPOCH(?P<epoch>\d+)$"
)


def build_model_name(base_model_name: str, data_set: str, zcp: str, epoch: int | str) -> str:
    if epoch == 0:
        return base_model_name
    return f"{base_model_name}_finetuned_DATASET{data_set}_ZCP{zcp}_EPOCH{epoch}"


def parse_finetuned_model_name(llm_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    match = _FINETUNED_NAME_PATTERN.search(llm_name)
    if not match:
        return None, None, None
    return match.group("data_set"), match.group("zcp"), match.group("epoch")

