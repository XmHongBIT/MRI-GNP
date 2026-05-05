from pathlib import Path

import yaml


def _load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected a mapping at {path}.")
    return data


def _resolve_path(config_path, maybe_relative):
    if maybe_relative in [None, ""]:
        return maybe_relative
    path = Path(maybe_relative)
    if path.is_absolute():
        return str(path)
    return str((Path(config_path).resolve().parent / path).resolve())


def load_configs(global_config_path, experiment_config_path):
    global_config_path = str(Path(global_config_path).resolve())
    experiment_config_path = str(Path(experiment_config_path).resolve())

    global_config = _load_yaml(global_config_path)
    experiment_config = _load_yaml(experiment_config_path)

    global_config.setdefault("seed", 42)
    global_config.setdefault("output_root_dir", "outputs")
    global_config.setdefault("num_workers", 4)
    global_config.setdefault("pin_memory", True)
    global_config.setdefault("use_amp", True)
    global_config.setdefault("save_predictions", True)
    global_config.setdefault("save_relation_summary", True)

    global_config["output_root_dir"] = _resolve_path(
        global_config_path,
        global_config["output_root_dir"],
    )

    manifest_path = experiment_config.get("manifest_path")
    if manifest_path:
        experiment_config["manifest_path"] = _resolve_path(experiment_config_path, manifest_path)

    experiment_config.setdefault("name", Path(experiment_config_path).stem)
    experiment_config.setdefault("sample_path_column", "sample_path")
    experiment_config.setdefault("split_column", "split")
    experiment_config.setdefault("train_split", "train")
    experiment_config.setdefault("val_split", "val")
    experiment_config.setdefault("test_split", "test")
    experiment_config.setdefault("subject_name_column", "subject_name")
    experiment_config.setdefault("data_source_column", "data_source")
    experiment_config.setdefault("image_key", "image")
    experiment_config.setdefault("meta_key", "meta")
    experiment_config.setdefault("num_epochs", experiment_config.get("num_epochs_for_training", 20))
    experiment_config.setdefault("batch_size", experiment_config.get("fast_batch_size", 16))
    experiment_config.setdefault("train_sampling_strategy", "weighted_sampler")
    experiment_config.setdefault("validate_every_n_epochs", 1)
    experiment_config.setdefault("checkpoint_every_n_epochs", 1)

    optimizer_cfg = experiment_config.setdefault("optimizer", {})
    optimizer_cfg.setdefault("lr", experiment_config.get("fast_learning_rate", 4e-5))
    optimizer_cfg.setdefault("weight_decay", experiment_config.get("fast_weight_decay", 1e-4))

    model_cfg = experiment_config.setdefault("model", {})
    model_cfg.setdefault("backbone", "vit")
    model_cfg.setdefault("image_size", experiment_config.get("fast_image_size", 224))
    model_cfg.setdefault("patch_size", 32)
    model_cfg.setdefault("hidden_size", 768)
    model_cfg.setdefault("num_hidden_layers", 12)
    model_cfg.setdefault("num_attention_heads", 12)
    model_cfg.setdefault("intermediate_size", 3072)
    model_cfg.setdefault("hidden_dropout_prob", 0.0)
    model_cfg.setdefault("attention_probs_dropout_prob", 0.0)
    model_cfg.setdefault("shared_backbone_layers", experiment_config.get("shared_backbone_layers", 6))
    model_cfg.setdefault("task_hidden_dim", experiment_config.get("multitask_task_hidden_dim", 256))
    model_cfg.setdefault("pretrained_source", experiment_config.get("vit_pretrained_model_path", ""))
    model_cfg["pretrained_source"] = _resolve_path(
        experiment_config_path,
        model_cfg.get("pretrained_source", ""),
    )

    relation_cfg = experiment_config.setdefault("relation", {})
    relation_cfg.setdefault(
        "enabled",
        experiment_config.get(
            "enable_hierarchical_relation_learning",
            experiment_config.get("enable_task_relation_learning", True),
        ),
    )
    relation_cfg.setdefault("num_layers", experiment_config.get("task_relation_num_layers", 2))
    relation_cfg.setdefault("num_heads", experiment_config.get("task_relation_num_heads", 4))
    relation_cfg.setdefault("dropout", experiment_config.get("task_relation_dropout", 0.1))
    relation_cfg.setdefault("ffn_mult", experiment_config.get("task_relation_ffn_mult", 2.0))
    relation_cfg.setdefault(
        "use_branch_embeddings",
        experiment_config.get(
            "task_relation_use_branch_embeddings",
            experiment_config.get("task_relation_use_task_embeddings", True),
        ),
    )
    relation_cfg.setdefault(
        "residual_scale_init",
        experiment_config.get("task_relation_residual_scale_init", 0.0),
    )

    experiment_config["_global_config_path"] = global_config_path
    experiment_config["_experiment_config_path"] = experiment_config_path
    return global_config, experiment_config
