import csv
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import (
    NpzManifestDataset,
    filter_rows_with_any_valid_label,
    load_manifest,
    normalize_hierarchy,
    normalize_task_specs,
    split_manifest,
    build_multitask_sampler,
)
from .metrics import (
    compute_classification_metrics,
    finite_row_mask,
    ordinal_pred_from_logits,
    ordinal_probabilities_tensor,
    ordinal_targets,
)
from .model import HierarchicalRelationMultiTaskClassifier, unwrap_model
from .utils import ensure_dir, save_json, set_seed, to_plain_types


def resolve_device(global_config, gpu_override=None):
    if torch.cuda.is_available():
        gpu_id = gpu_override if gpu_override is not None else int(global_config.get("run_on_which_gpu", 0))
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def build_task_criteria(task_specs, train_label_matrix, device):
    criteria = {}
    for task_id, task_spec in enumerate(task_specs):
        labels = train_label_matrix[:, task_id]
        valid_labels = labels[labels >= 0]
        if task_spec.loss_type == "ordinal":
            num_classes = len(task_spec.output_classes)
            if task_spec.ordinal_use_pos_weight and len(valid_labels) > 0:
                pos_weights = []
                for threshold_id in range(num_classes - 1):
                    positives = int((valid_labels > threshold_id).sum())
                    negatives = max(1, len(valid_labels) - positives)
                    pos_weights.append(negatives / max(1, positives))
                criteria[task_spec.name] = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(pos_weights, dtype=torch.float32, device=device)
                )
            else:
                criteria[task_spec.name] = nn.BCEWithLogitsLoss()
        else:
            if task_spec.use_class_weight and len(valid_labels) > 0:
                class_counts = [max(1, int((valid_labels == class_id).sum())) for class_id in range(len(task_spec.output_classes))]
                class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float32, device=device)
                class_weights = class_weights / class_weights.sum() * len(class_weights)
                criteria[task_spec.name] = nn.CrossEntropyLoss(weight=class_weights)
            else:
                criteria[task_spec.name] = nn.CrossEntropyLoss()
    return criteria


def append_nonfinite_log(log_path: Optional[Path], phase_name: str, task_name: str, subject_names: List[str], data_sources: List[str], issue: str):
    if log_path is None or not subject_names:
        return
    with Path(log_path).open("a", encoding="utf-8") as f:
        for subject_name, data_source in zip(subject_names, data_sources):
            f.write(f"[{phase_name}] task={task_name} subject={subject_name} data_source={data_source} issue={issue}\n")


def compute_task_metrics(task_spec, total_loss, total_count, y_true, y_pred, y_logits=None):
    num_classes = len(task_spec.output_classes)
    if task_spec.loss_type == "ordinal":
        y_probs = None
        if y_logits is not None and len(y_logits) > 0:
            logits = torch.cat(y_logits, dim=0).float()
            y_probs = ordinal_probabilities_tensor(logits, num_classes).detach().cpu().numpy()
        metrics = compute_classification_metrics(total_loss, total_count, y_true, y_pred, num_classes, y_probs=y_probs)
        if y_true:
            metrics["mae"] = float(np.mean(np.abs(np.asarray(y_true, dtype="float32") - np.asarray(y_pred, dtype="float32"))))
        else:
            metrics["mae"] = 0.0
        return metrics

    y_probs = None
    if y_logits is not None and len(y_logits) > 0:
        logits = torch.cat(y_logits, dim=0).float()
        y_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return compute_classification_metrics(total_loss, total_count, y_true, y_pred, num_classes, y_probs=y_probs)


def run_one_epoch_multitask(
    model,
    loader,
    optimizer,
    task_criteria,
    task_specs,
    device,
    scaler,
    train,
    use_amp,
    phase_name="train",
    nonfinite_log_path=None,
):
    model.train(train)
    task_trackers = {
        task_spec.name: {"loss": 0.0, "count": 0, "y_true": [], "y_pred": [], "y_logits": []}
        for task_spec in task_specs
    }

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)
        task_labels = batch["task_labels"].to(device, non_blocking=True)
        task_masks = batch["task_masks"].to(device, non_blocking=True)
        subject_names = list(batch["subject_name"])
        data_sources = list(batch["data_source"])

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                outputs = model.extract_feature_dict(image, meta)
                active_losses = []
                for task_id, task_spec in enumerate(task_specs):
                    valid_mask = task_masks[:, task_id]
                    if torch.sum(valid_mask).item() == 0:
                        continue
                    logits = outputs[task_spec.name]["logits"][valid_mask]
                    labels = task_labels[valid_mask, task_id]
                    sample_indices = torch.where(valid_mask)[0].detach().cpu().tolist()
                    finite_mask = finite_row_mask(logits)
                    finite_mask_list = finite_mask.detach().cpu().tolist()
                    if not bool(torch.all(finite_mask).item()):
                        invalid_rows = [row_id for row_id, is_valid in enumerate(finite_mask_list) if not is_valid]
                        append_nonfinite_log(
                            nonfinite_log_path,
                            phase_name,
                            task_spec.name,
                            [subject_names[sample_indices[row_id]] for row_id in invalid_rows],
                            [data_sources[sample_indices[row_id]] for row_id in invalid_rows],
                            "non-finite logits",
                        )
                        if not bool(torch.any(finite_mask).item()):
                            continue
                        logits = logits[finite_mask]
                        labels = labels[finite_mask]
                        sample_indices = [sample_indices[row_id] for row_id, is_valid in enumerate(finite_mask_list) if is_valid]
                    if task_spec.loss_type == "ordinal":
                        task_loss = task_criteria[task_spec.name](logits, ordinal_targets(labels, len(task_spec.output_classes)))
                        preds = ordinal_pred_from_logits(logits)
                    else:
                        task_loss = task_criteria[task_spec.name](logits, labels)
                        preds = torch.argmax(logits, dim=1)
                    if not bool(torch.isfinite(task_loss).item()):
                        append_nonfinite_log(
                            nonfinite_log_path,
                            phase_name,
                            task_spec.name,
                            [subject_names[sample_id] for sample_id in sample_indices],
                            [data_sources[sample_id] for sample_id in sample_indices],
                            "non-finite task loss",
                        )
                        continue
                    active_losses.append(task_spec.task_weight * task_loss)
                    batch_size = int(labels.shape[0])
                    task_trackers[task_spec.name]["loss"] += float(task_loss.item()) * batch_size
                    task_trackers[task_spec.name]["count"] += batch_size
                    task_trackers[task_spec.name]["y_true"].extend(labels.detach().cpu().tolist())
                    task_trackers[task_spec.name]["y_pred"].extend(preds.detach().cpu().tolist())
                    task_trackers[task_spec.name]["y_logits"].append(logits.detach().float().cpu())
                total_loss = torch.stack(active_losses).sum() / max(1, len(active_losses)) if active_losses else None
            if total_loss is not None and not bool(torch.isfinite(total_loss).item()):
                append_nonfinite_log(nonfinite_log_path, phase_name, "_overall", subject_names, data_sources, "non-finite total loss")
                total_loss = None
            if train and total_loss is not None:
                optimizer.zero_grad(set_to_none=True)
                if use_amp and scaler is not None:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

    metrics = {}
    weighted_auc = 0.0
    weighted_accuracy = 0.0
    active_task_count = 0
    for task_spec in task_specs:
        tracker = task_trackers[task_spec.name]
        metrics[task_spec.name] = compute_task_metrics(
            task_spec,
            tracker["loss"],
            tracker["count"],
            tracker["y_true"],
            tracker["y_pred"],
            tracker["y_logits"] if tracker["y_logits"] else None,
        )
        if tracker["count"] > 0:
            task_auc = metrics[task_spec.name].get("auc")
            if task_auc is None:
                task_auc = metrics[task_spec.name].get("balanced_accuracy", 0.0)
            weighted_auc += float(task_auc) * float(task_spec.task_weight)
            weighted_accuracy += float(metrics[task_spec.name].get("accuracy", 0.0)) * float(task_spec.task_weight)
            active_task_count += 1
    metrics["_overall"] = {
        "macro_auc": weighted_auc / max(1, active_task_count),
        "macro_accuracy": weighted_accuracy / max(1, active_task_count),
        "active_tasks": active_task_count,
    }
    return metrics


def save_predictions(model, loader, device, task_specs, output_dir, split_name, use_amp):
    model.eval()
    storage = {task_spec.name: [] for task_spec in task_specs}
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            task_labels = batch["task_labels"].to(device, non_blocking=True)
            task_masks = batch["task_masks"].to(device, non_blocking=True)
            subject_names = list(batch["subject_name"])
            data_sources = list(batch["data_source"])
            sample_paths = list(batch["sample_path"])
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                outputs = model.extract_feature_dict(image, meta)
            for task_id, task_spec in enumerate(task_specs):
                valid_mask = task_masks[:, task_id]
                if torch.sum(valid_mask).item() == 0:
                    continue
                valid_indices = torch.where(valid_mask)[0]
                logits = outputs[task_spec.name]["logits"][valid_indices].float()
                labels = task_labels[valid_indices, task_id]
                if task_spec.loss_type == "ordinal":
                    probs = ordinal_probabilities_tensor(logits, len(task_spec.output_classes))
                    preds = ordinal_pred_from_logits(logits)
                else:
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                probs_cpu = probs.detach().cpu().numpy().astype("float32")
                preds_cpu = preds.detach().cpu().numpy().astype("int64")
                labels_cpu = labels.detach().cpu().numpy().astype("int64")
                for local_id, sample_id in enumerate(valid_indices.detach().cpu().tolist()):
                    row = {
                        "subject_name": subject_names[sample_id],
                        "data_source": data_sources[sample_id],
                        "sample_path": sample_paths[sample_id],
                        "pred_class": task_spec.output_classes[int(preds_cpu[local_id])],
                        "true_label": task_spec.output_classes[int(labels_cpu[local_id])],
                    }
                    for class_id, class_name in enumerate(task_spec.output_classes):
                        row[f"prob_{class_name}"] = float(probs_cpu[local_id, class_id])
                    storage[task_spec.name].append(row)
    for task_spec in task_specs:
        rows = storage[task_spec.name]
        if not rows:
            continue
        csv_path = Path(output_dir) / f"multitask_{task_spec.name}_{split_name}_predictions.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def save_relation_summary(model, output_dir):
    raw_model = unwrap_model(model)
    summary = {
        "task_names": list(raw_model.task_names),
        "enable_hierarchical_relation_learning": bool(getattr(raw_model, "enable_hierarchical_relation_learning", False)),
    }
    router = getattr(raw_model, "hierarchy_router", None)
    if router is not None:
        summary.update(router.summarize())
    save_json(summary, Path(output_dir) / "multitask_relation_summary.json")


def _describe_split(name, rows, task_specs):
    summary = {"num_samples": len(rows)}
    for task_spec in task_specs:
        counts = {class_name: 0 for class_name in task_spec.output_classes}
        labeled = 0
        for row in rows:
            value = row.get(task_spec.label_column, "")
            mapped = value
            if task_spec.output_classes_mapping is not None:
                from .data import map_label_with_rules
                mapped = map_label_with_rules(value, task_spec.output_classes_mapping)
            if mapped in counts:
                counts[mapped] += 1
                labeled += 1
        summary[task_spec.name] = {"labeled": labeled, "counts": counts}
    print(f"{name} split: {summary}")


def train_pipeline(global_config, experiment_config, gpu_override=None, dry_run=False):
    set_seed(int(global_config.get("seed", 42)))
    output_dir = ensure_dir(Path(global_config["output_root_dir"]) / experiment_config["name"])
    save_json(to_plain_types(global_config), output_dir / "global_config.snapshot.json")
    save_json(to_plain_types(experiment_config), output_dir / "experiment_config.snapshot.json")

    raw_task_specs = experiment_config.get("tasks", experiment_config.get("multitask_tasks"))
    if not raw_task_specs:
        raise RuntimeError('Expected "tasks" in experiment config.')
    task_specs = normalize_task_specs(raw_task_specs)
    raw_hierarchy = experiment_config.get("hierarchy", experiment_config.get("multitask_hierarchy"))
    hierarchy = normalize_hierarchy(raw_hierarchy, task_specs)

    manifest_rows = load_manifest(experiment_config["manifest_path"])
    manifest_dir = Path(experiment_config["manifest_path"]).resolve().parent
    sample_path_column = experiment_config["sample_path_column"]
    for row in manifest_rows:
        raw_path = row.get(sample_path_column, "")
        if not raw_path:
            continue
        sample_path = Path(raw_path)
        if not sample_path.is_absolute():
            row[sample_path_column] = str((manifest_dir / sample_path).resolve())
    manifest_rows = filter_rows_with_any_valid_label(manifest_rows, task_specs)
    train_rows, val_rows, test_rows = split_manifest(
        manifest_rows,
        experiment_config["split_column"],
        experiment_config["train_split"],
        experiment_config["val_split"],
        experiment_config["test_split"],
    )
    if not train_rows:
        raise RuntimeError("No training samples found after filtering.")

    _describe_split("train", train_rows, task_specs)
    _describe_split("val", val_rows, task_specs)
    _describe_split("test", test_rows, task_specs)

    dataset_kwargs = {
        "task_specs": task_specs,
        "sample_path_column": experiment_config["sample_path_column"],
        "image_key": experiment_config["image_key"],
        "meta_key": experiment_config["meta_key"],
        "subject_name_column": experiment_config["subject_name_column"],
        "data_source_column": experiment_config["data_source_column"],
    }
    train_dataset = NpzManifestDataset(
        rows=train_rows,
        train=True,
        enable_augmentation=bool(experiment_config.get("enable_train_image_augmentation", True)),
        **dataset_kwargs,
    )
    val_dataset = NpzManifestDataset(rows=val_rows, train=False, enable_augmentation=False, **dataset_kwargs) if val_rows else None
    test_dataset = NpzManifestDataset(rows=test_rows, train=False, enable_augmentation=False, **dataset_kwargs) if test_rows else None

    train_sampler = build_multitask_sampler(train_dataset.label_matrix.numpy(), experiment_config.get("train_sampling_strategy", "weighted_sampler"))
    num_workers = int(global_config.get("num_workers", 4))
    loader_kwargs = {
        "batch_size": int(experiment_config.get("batch_size", 16)),
        "num_workers": num_workers,
        "pin_memory": bool(global_config.get("pin_memory", True)),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=train_sampler is None, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs) if test_dataset is not None else None

    device = resolve_device(global_config, gpu_override=gpu_override)
    model = HierarchicalRelationMultiTaskClassifier(
        in_channels=train_dataset.in_channels,
        meta_dim=train_dataset.meta_dim,
        task_specs=task_specs,
        hierarchy=hierarchy,
        model_config=experiment_config["model"],
        relation_config=experiment_config["relation"],
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True

    if dry_run:
        batch = next(iter(train_loader))
        image = batch["image"].to(device)
        meta = batch["meta"].to(device)
        with torch.no_grad():
            outputs = model.extract_feature_dict(image, meta)
        summary = {
            "device": str(device),
            "batch_size": int(image.shape[0]),
            "image_shape": list(image.shape),
            "meta_shape": list(meta.shape),
            "tasks": {task_name: list(task_output["logits"].shape) for task_name, task_output in outputs.items()},
        }
        print(summary)
        save_json(summary, output_dir / "dry_run_summary.json")
        return

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(experiment_config["optimizer"]["lr"]),
        weight_decay=float(experiment_config["optimizer"]["weight_decay"]),
    )
    use_amp = bool(global_config.get("use_amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    task_criteria = build_task_criteria(task_specs, train_dataset.label_matrix.numpy(), device)

    best_metric = -1.0
    history = []
    latest_path = output_dir / "multitask_model_latest.pt"
    best_path = output_dir / "multitask_model_best.pt"
    metrics_path = output_dir / "multitask_metrics.json"
    nonfinite_log_path = output_dir / "nonfinite_sample.log"
    if nonfinite_log_path.exists():
        nonfinite_log_path.unlink()

    for epoch in range(1, int(experiment_config.get("num_epochs", 20)) + 1):
        train_metrics = run_one_epoch_multitask(
            model,
            train_loader,
            optimizer,
            task_criteria,
            task_specs,
            device,
            scaler,
            True,
            use_amp,
            phase_name="train",
            nonfinite_log_path=nonfinite_log_path,
        )
        val_metrics = None
        if val_loader is not None and (
            int(experiment_config.get("validate_every_n_epochs", 1)) <= 1
            or epoch % int(experiment_config.get("validate_every_n_epochs", 1)) == 0
            or epoch == int(experiment_config.get("num_epochs", 20))
        ):
            val_metrics = run_one_epoch_multitask(
                model,
                val_loader,
                optimizer,
                task_criteria,
                task_specs,
                device,
                scaler,
                False,
                use_amp,
                phase_name="val",
                nonfinite_log_path=nonfinite_log_path,
            )

        record = {"epoch": epoch, "train_overall": train_metrics["_overall"]}
        for task_spec in task_specs:
            record[f"train_{task_spec.name}"] = train_metrics[task_spec.name]
        if val_metrics is not None:
            record["val_overall"] = val_metrics["_overall"]
            for task_spec in task_specs:
                record[f"val_{task_spec.name}"] = val_metrics[task_spec.name]
        history.append(record)

        msg = f"epoch {epoch} | train macro_auc {train_metrics['_overall']['macro_auc']:.4f} acc {train_metrics['_overall']['macro_accuracy']:.4f}"
        if val_metrics is not None:
            msg += f" | val macro_auc {val_metrics['_overall']['macro_auc']:.4f} acc {val_metrics['_overall']['macro_accuracy']:.4f}"
        print(msg)

        current_metrics = val_metrics if val_metrics is not None else train_metrics
        current_metric = float(current_metrics["_overall"]["macro_auc"]) + float(current_metrics["_overall"]["macro_accuracy"])
        should_checkpoint = (
            epoch % max(1, int(experiment_config.get("checkpoint_every_n_epochs", 1))) == 0
            or epoch == int(experiment_config.get("num_epochs", 20))
        )
        if should_checkpoint:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "history": history},
                latest_path,
            )
        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "history": history},
                best_path,
            )
        save_json({"history": history, "best_metric": best_metric}, metrics_path)

    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model"])

    final_metrics = {"history": history, "best_metric": best_metric}
    if val_loader is not None:
        final_metrics["val"] = run_one_epoch_multitask(
            model,
            val_loader,
            optimizer,
            task_criteria,
            task_specs,
            device,
            scaler,
            False,
            use_amp,
            phase_name="val",
            nonfinite_log_path=nonfinite_log_path,
        )
        if bool(global_config.get("save_predictions", True)):
            save_predictions(model, val_loader, device, task_specs, output_dir, "val", use_amp)
    if test_loader is not None:
        final_metrics["test"] = run_one_epoch_multitask(
            model,
            test_loader,
            optimizer,
            task_criteria,
            task_specs,
            device,
            scaler,
            False,
            use_amp,
            phase_name="test",
            nonfinite_log_path=nonfinite_log_path,
        )
        if bool(global_config.get("save_predictions", True)):
            save_predictions(model, test_loader, device, task_specs, output_dir, "test", use_amp)
    if bool(global_config.get("save_relation_summary", True)):
        save_relation_summary(model, output_dir)
    save_json(final_metrics, output_dir / "multitask_final_metrics.json")
    print(f"finished. output dir: {output_dir}")
