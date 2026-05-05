import csv
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


@dataclass
class TaskSpec:
    name: str
    label_column: str
    output_classes: List[str]
    output_classes_mapping: Optional[dict]
    loss_type: str = "ce"
    task_weight: float = 1.0
    use_class_weight: bool = True
    ordinal_use_pos_weight: bool = True


@dataclass
class BranchSpec:
    name: str
    tasks: List[str]
    include_root: bool = True


def normalize_task_specs(raw_task_specs: List[dict]) -> List[TaskSpec]:
    task_specs = [
        TaskSpec(
            name=str(raw["name"]),
            label_column=str(raw.get("label_column", raw.get("output"))),
            output_classes=[str(item) for item in raw["output_classes"]],
            output_classes_mapping=copy.deepcopy(raw.get("output_classes_mapping")),
            loss_type=str(raw.get("loss_type", "ce")).lower(),
            task_weight=float(raw.get("task_weight", 1.0)),
            use_class_weight=bool(raw.get("use_class_weight", True)),
            ordinal_use_pos_weight=bool(raw.get("ordinal_use_pos_weight", True)),
        )
        for raw in raw_task_specs
    ]
    task_names = [task_spec.name for task_spec in task_specs]
    if len(task_names) != len(set(task_names)):
        raise RuntimeError("Every task name must be unique.")
    return task_specs


def default_hierarchy(task_specs: List[TaskSpec]) -> dict:
    task_names = [task_spec.name for task_spec in task_specs]
    root_task = "IDH" if "IDH" in task_names else task_names[0]
    branches = []
    if "1p19q" in task_names or "ATRX" in task_names or "TERT" in task_names:
        oligo_tasks = [task_name for task_name in ["1p19q", "ATRX", "TERT"] if task_name in task_names]
        if oligo_tasks:
            branches.append({"name": "oligo", "tasks": oligo_tasks, "include_root": True})
    if "ATRX" in task_names or "TP53" in task_names or "CDKN" in task_names:
        astro_tasks = [task_name for task_name in ["ATRX", "TP53", "CDKN"] if task_name in task_names]
        if astro_tasks:
            branches.append({"name": "astro", "tasks": astro_tasks, "include_root": True})
    if "TERT" in task_names or "EGFR" in task_names or "+7-10" in task_names:
        gbm_tasks = [task_name for task_name in ["TERT", "EGFR", "+7-10"] if task_name in task_names]
        if gbm_tasks:
            branches.append({"name": "gbm", "tasks": gbm_tasks, "include_root": True})
    return {"root_task": root_task, "branches": branches}


def normalize_hierarchy(raw_hierarchy: Optional[dict], task_specs: List[TaskSpec]) -> dict:
    hierarchy = copy.deepcopy(raw_hierarchy) if raw_hierarchy is not None else default_hierarchy(task_specs)
    if not isinstance(hierarchy, dict):
        raise RuntimeError("Hierarchy must be a dict.")
    task_names = [task_spec.name for task_spec in task_specs]
    root_task = str(hierarchy.get("root_task", task_names[0]))
    if root_task not in task_names:
        raise RuntimeError(f'Hierarchy root_task "{root_task}" is not in tasks.')
    branches: List[BranchSpec] = []
    branch_names = []
    for raw_branch in hierarchy.get("branches", []):
        branch = BranchSpec(
            name=str(raw_branch["name"]),
            tasks=[
                str(task_name)
                for task_name in raw_branch.get("tasks", [])
                if str(task_name) in task_names and str(task_name) != root_task
            ],
            include_root=bool(raw_branch.get("include_root", True)),
        )
        if not branch.tasks:
            continue
        branch_names.append(branch.name)
        branches.append(branch)
    if len(branch_names) != len(set(branch_names)):
        raise RuntimeError("Every hierarchy branch name must be unique.")
    task_to_branches = {task_name: [] for task_name in task_names}
    for branch in branches:
        for task_name in branch.tasks:
            task_to_branches[task_name].append(branch.name)
    return {
        "root_task": root_task,
        "branches": branches,
        "task_to_branches": task_to_branches,
    }


def map_label_with_rules(raw_value, mapping_dict):
    value = "" if raw_value is None else str(raw_value).strip()
    if mapping_dict is None:
        return value
    for mapped_value, source_values in mapping_dict.items():
        if not isinstance(source_values, list):
            source_values = [source_values]
        if value in [str(item).strip() for item in source_values]:
            return str(mapped_value)
    return value


def load_manifest(path: str) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if row is None:
                continue
            normalized = {str(key): ("" if value is None else str(value).strip()) for key, value in row.items()}
            if not any(normalized.values()):
                continue
            rows.append(normalized)
    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")
    return rows


def split_manifest(rows: List[dict], split_column: str, train_split: str, val_split: str, test_split: str):
    train_rows, val_rows, test_rows = [], [], []
    for row in rows:
        split_value = row.get(split_column, "")
        if split_value == train_split:
            train_rows.append(row)
        elif split_value == val_split:
            val_rows.append(row)
        elif split_value == test_split:
            test_rows.append(row)
    return train_rows, val_rows, test_rows


def filter_rows_with_any_valid_label(rows: List[dict], task_specs: List[TaskSpec]) -> List[dict]:
    filtered = []
    for row in rows:
        for task_spec in task_specs:
            mapped = map_label_with_rules(row.get(task_spec.label_column, ""), task_spec.output_classes_mapping)
            if mapped in task_spec.output_classes:
                filtered.append(row)
                break
    return filtered


def build_label_matrix(rows: List[dict], task_specs: List[TaskSpec]) -> np.ndarray:
    task_name_to_id = {
        task_spec.name: {class_name: idx for idx, class_name in enumerate(task_spec.output_classes)}
        for task_spec in task_specs
    }
    matrix = np.full((len(rows), len(task_specs)), fill_value=-1, dtype="int64")
    for row_id, row in enumerate(rows):
        for task_id, task_spec in enumerate(task_specs):
            mapped = map_label_with_rules(row.get(task_spec.label_column, ""), task_spec.output_classes_mapping)
            if mapped in task_name_to_id[task_spec.name]:
                matrix[row_id, task_id] = task_name_to_id[task_spec.name][mapped]
    return matrix


def build_multitask_sampler(label_matrix: np.ndarray, strategy: str):
    if strategy == "none":
        return None
    if strategy != "weighted_sampler":
        raise RuntimeError(f'Unsupported train_sampling_strategy: "{strategy}".')
    if label_matrix.shape[0] == 0:
        return None
    sample_weights = np.zeros((label_matrix.shape[0],), dtype="float32")
    for task_id in range(label_matrix.shape[1]):
        labels = label_matrix[:, task_id]
        valid_mask = labels >= 0
        if int(np.sum(valid_mask)) == 0:
            continue
        counts = {}
        for label in labels[valid_mask].tolist():
            counts[label] = counts.get(label, 0) + 1
        for sample_id, label in enumerate(labels.tolist()):
            if label >= 0:
                sample_weights[sample_id] += 1.0 / counts[label]
    zero_mask = sample_weights <= 0
    if bool(np.all(zero_mask)):
        return None
    sample_weights[zero_mask] = sample_weights[~zero_mask].min()
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights.tolist()),
        num_samples=int(label_matrix.shape[0]),
        replacement=True,
    )


class NpzManifestDataset(Dataset):
    def __init__(
        self,
        rows: List[dict],
        task_specs: List[TaskSpec],
        sample_path_column: str,
        image_key: str,
        meta_key: str,
        subject_name_column: str,
        data_source_column: str,
        train: bool = False,
        enable_augmentation: bool = False,
    ):
        self.rows = rows
        self.task_specs = task_specs
        self.sample_path_column = sample_path_column
        self.image_key = image_key
        self.meta_key = meta_key
        self.subject_name_column = subject_name_column
        self.data_source_column = data_source_column
        self.train = train
        self.enable_augmentation = enable_augmentation
        self.label_matrix = torch.as_tensor(build_label_matrix(rows, task_specs), dtype=torch.long)
        self.in_channels, self.meta_dim = self._probe_sample_shapes()

    def _probe_sample_shapes(self):
        for row in self.rows:
            sample_path = Path(row[self.sample_path_column])
            if not sample_path.exists():
                raise RuntimeError(f"Sample file not found: {sample_path}")
            with np.load(sample_path, allow_pickle=False) as data:
                if self.image_key not in data:
                    raise RuntimeError(f'"{self.image_key}" not found in sample: {sample_path}')
                image = np.asarray(data[self.image_key], dtype="float32")
                if image.ndim == 2:
                    image = image[None, ...]
                if image.ndim != 3:
                    raise RuntimeError(f"Expected image shape [C,H,W], got {image.shape} in {sample_path}")
                if self.meta_key in data:
                    meta = np.asarray(data[self.meta_key], dtype="float32").reshape(-1)
                else:
                    meta = np.zeros((0,), dtype="float32")
                return int(image.shape[0]), int(meta.shape[0])
        raise RuntimeError("Could not probe dataset sample shapes.")

    def __len__(self):
        return len(self.rows)

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[1])
        if torch.rand(1).item() < 0.5:
            image = torch.rot90(image, k=int(torch.randint(0, 4, (1,)).item()), dims=[1, 2])
        if torch.rand(1).item() < 0.3:
            image = image + 0.02 * torch.randn_like(image)
        return image

    def __getitem__(self, index):
        row = self.rows[index]
        sample_path = Path(row[self.sample_path_column])
        with np.load(sample_path, allow_pickle=False) as data:
            image = np.asarray(data[self.image_key], dtype="float32")
            if image.ndim == 2:
                image = image[None, ...]
            meta = np.asarray(data[self.meta_key], dtype="float32").reshape(-1) if self.meta_key in data else np.zeros((self.meta_dim,), dtype="float32")
        image_tensor = torch.from_numpy(image)
        meta_tensor = torch.from_numpy(meta)
        if self.train and self.enable_augmentation:
            image_tensor = self._augment(image_tensor)
        return {
            "image": image_tensor,
            "meta": meta_tensor,
            "task_labels": self.label_matrix[index],
            "task_masks": self.label_matrix[index] >= 0,
            "subject_name": row.get(self.subject_name_column, sample_path.stem),
            "data_source": row.get(self.data_source_column, ""),
            "sample_path": str(sample_path),
        }
