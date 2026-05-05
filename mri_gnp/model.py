import copy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .data import BranchSpec, TaskSpec

try:
    from torchvision.models import resnet18
except Exception:
    resnet18 = None

try:
    from transformers.models.vit.configuration_vit import ViTConfig
    from transformers.models.vit.modeling_vit import ViTModel
except Exception:
    ViTConfig = None
    ViTModel = None


class SmallBackbone(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out_dim = 256

    def forward(self, x):
        return self.net(x).flatten(1)


class CrossTaskRelationBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, ffn_mult: float, residual_scale_init: float):
        super().__init__()
        hidden_dim = max(embed_dim, int(round(embed_dim * ffn_mult)))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.attn_scale = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))
        self.ffn_scale = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))

    def forward(self, task_embeddings: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(task_embeddings)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input, need_weights=False)
        task_embeddings = task_embeddings + torch.tanh(self.attn_scale) * self.dropout1(attn_output)
        ffn_output = self.ffn(self.norm2(task_embeddings))
        task_embeddings = task_embeddings + torch.tanh(self.ffn_scale) * self.dropout2(ffn_output)
        return task_embeddings


class HierarchicalRelationRouter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hierarchy: dict,
        num_layers: int,
        num_heads: int,
        dropout: float,
        ffn_mult: float,
        use_branch_embeddings: bool,
        residual_scale_init: float,
    ):
        super().__init__()
        self.root_task = hierarchy["root_task"]
        self.branches: List[BranchSpec] = hierarchy["branches"]
        self.task_to_branches = hierarchy["task_to_branches"]
        self.branch_names = [branch.name for branch in self.branches]
        self.enable_router = len(self.branches) > 0
        self.branch_embeddings = nn.ParameterDict()
        self.branch_projectors = nn.ModuleDict()
        self.branch_root_blocks = nn.ModuleDict()
        self.branch_task_blocks = nn.ModuleDict()
        self.task_fusers = nn.ModuleDict()
        if not self.enable_router:
            return
        if embed_dim % int(num_heads) != 0:
            raise RuntimeError("task_hidden_dim must be divisible by relation num_heads.")
        for branch in self.branches:
            if bool(use_branch_embeddings):
                self.branch_embeddings[branch.name] = nn.Parameter(torch.zeros((embed_dim,), dtype=torch.float32))
                nn.init.normal_(self.branch_embeddings[branch.name], mean=0.0, std=0.02)
            self.branch_projectors[branch.name] = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.branch_root_blocks[branch.name] = nn.ModuleList(
                [
                    CrossTaskRelationBlock(embed_dim, num_heads, dropout, ffn_mult, residual_scale_init)
                    for _ in range(max(1, int(num_layers)))
                ]
            )
            self.branch_task_blocks[branch.name] = nn.ModuleList(
                [
                    CrossTaskRelationBlock(embed_dim, num_heads, dropout, ffn_mult, residual_scale_init)
                    for _ in range(max(1, int(num_layers)))
                ]
            )
        all_tasks = set(self.task_to_branches.keys()) | {self.root_task}
        self.task_fusers = nn.ModuleDict({task_name: nn.Linear(embed_dim, 1) for task_name in all_tasks})

    def _build_branch_context(self, branch: BranchSpec, base_task_features: Dict[str, torch.Tensor], root_feature: torch.Tensor):
        member_features = torch.stack([base_task_features[task_name] for task_name in branch.tasks], dim=1)
        member_mean = member_features.mean(dim=1)
        root_input = root_feature if branch.include_root else torch.zeros_like(root_feature)
        branch_seed = member_mean
        if branch.name in self.branch_embeddings:
            branch_seed = branch_seed + self.branch_embeddings[branch.name].unsqueeze(0).to(member_mean.dtype)
        branch_context = self.branch_projectors[branch.name](torch.cat([member_mean, root_input, branch_seed], dim=1))
        root_from_branch = root_feature
        if branch.include_root:
            root_tokens = torch.stack([branch_context, root_feature], dim=1)
            for block in self.branch_root_blocks[branch.name]:
                root_tokens = block(root_tokens)
            branch_context = root_tokens[:, 0, :]
            root_from_branch = root_tokens[:, 1, :]
        task_tokens = torch.cat([branch_context.unsqueeze(1), member_features], dim=1)
        for block in self.branch_task_blocks[branch.name]:
            task_tokens = block(task_tokens)
        refined_member_features = {
            task_name: task_tokens[:, task_id + 1, :]
            for task_id, task_name in enumerate(branch.tasks)
        }
        return task_tokens[:, 0, :], root_from_branch, refined_member_features

    def _fuse_candidates(self, task_name: str, candidates: List[torch.Tensor]) -> torch.Tensor:
        if len(candidates) == 1:
            return candidates[0]
        stacked = torch.stack(candidates, dim=1)
        gate_logits = self.task_fusers[task_name](stacked).squeeze(-1)
        gate_weights = torch.softmax(gate_logits, dim=1)
        return torch.sum(stacked * gate_weights.unsqueeze(-1), dim=1)

    def forward(self, base_task_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.enable_router:
            return {task_name: feature for task_name, feature in base_task_features.items()}
        root_feature = base_task_features[self.root_task]
        candidate_features = {task_name: [feature] for task_name, feature in base_task_features.items()}
        for branch in self.branches:
            _, root_from_branch, refined_member_features = self._build_branch_context(branch, base_task_features, root_feature)
            candidate_features[self.root_task].append(root_from_branch)
            for task_name, refined_feature in refined_member_features.items():
                candidate_features[task_name].append(refined_feature)
        return {
            task_name: self._fuse_candidates(task_name, candidates)
            for task_name, candidates in candidate_features.items()
        }

    def summarize(self) -> dict:
        summary = {
            "root_task": self.root_task,
            "branch_names": list(self.branch_names),
            "branch_include_root": {branch.name: bool(branch.include_root) for branch in self.branches},
            "task_to_branches": {task_name: list(branches) for task_name, branches in self.task_to_branches.items()},
        }
        if len(self.branch_embeddings) > 0:
            branch_names = list(self.branch_embeddings.keys())
            embeddings = np.stack([self.branch_embeddings[name].detach().cpu().numpy().astype("float32") for name in branch_names], axis=0)
            normalized = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), a_min=1e-8, a_max=None)
            summary["branch_embedding_cosine"] = np.matmul(normalized, normalized.T).round(6).tolist()
        summary["relation_residual_scales"] = {
            branch.name: {
                "root_blocks": [
                    {
                        "attn_scale": float(torch.tanh(block.attn_scale).detach().cpu().item()),
                        "ffn_scale": float(torch.tanh(block.ffn_scale).detach().cpu().item()),
                    }
                    for block in self.branch_root_blocks[branch.name]
                ],
                "task_blocks": [
                    {
                        "attn_scale": float(torch.tanh(block.attn_scale).detach().cpu().item()),
                        "ffn_scale": float(torch.tanh(block.ffn_scale).detach().cpu().item()),
                    }
                    for block in self.branch_task_blocks[branch.name]
                ],
            }
            for branch in self.branches
        }
        return summary


class HierarchicalRelationMultiTaskClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        meta_dim: int,
        task_specs: List[TaskSpec],
        hierarchy: dict,
        model_config: dict,
        relation_config: dict,
    ):
        super().__init__()
        if not task_specs:
            raise RuntimeError("At least one task is required.")
        self.task_specs = task_specs
        self.task_names = [task_spec.name for task_spec in task_specs]
        self.hierarchy = hierarchy
        self.backbone_name = str(model_config.get("backbone", "vit")).lower()
        self.use_vit = self.backbone_name == "vit" and ViTModel is not None and ViTConfig is not None
        self.channel_adapter = None

        if self.use_vit:
            base_backbone = self._build_vit_backbone(in_channels, model_config)
            self.embeddings = base_backbone.embeddings
            encoder_layers = list(base_backbone.encoder.layer)
            total_layers = len(encoder_layers)
            self.shared_backbone_layers = max(0, min(int(model_config.get("shared_backbone_layers", 6)), total_layers))
            self.shared_encoder_layers = nn.ModuleList(encoder_layers[: self.shared_backbone_layers])
            self.task_encoder_layers = nn.ModuleDict(
                {
                    task_name: nn.ModuleList(copy.deepcopy(encoder_layers[self.shared_backbone_layers:]))
                    for task_name in self.task_names
                }
            )
            self.task_layernorm = nn.ModuleDict(
                {task_name: copy.deepcopy(base_backbone.layernorm) for task_name in self.task_names}
            )
            self.task_pooler = nn.ModuleDict(
                {
                    task_name: copy.deepcopy(base_backbone.pooler) if base_backbone.pooler is not None else nn.Identity()
                    for task_name in self.task_names
                }
            )
            image_dim = int(model_config.get("hidden_size", 768))
        elif self.backbone_name == "resnet18" and resnet18 is not None:
            backbone = resnet18(weights=None)
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.shared_backbone_layers = 0
            image_dim = 512
        else:
            self.backbone = SmallBackbone(in_channels)
            self.shared_backbone_layers = 0
            image_dim = self.backbone.out_dim

        self.meta_head = None
        if meta_dim > 0:
            self.meta_head = nn.Sequential(
                nn.Linear(meta_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
            )
        fusion_dim = image_dim + (128 if self.meta_head is not None else 0)
        task_hidden_dim = int(model_config.get("task_hidden_dim", 256))
        self.task_necks = nn.ModuleDict(
            {
                task_name: nn.Sequential(
                    nn.Linear(fusion_dim, task_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                )
                for task_name in self.task_names
            }
        )
        self.enable_hierarchical_relation_learning = bool(relation_config.get("enabled", True)) and len(self.task_names) > 1
        self.hierarchy_router = (
            HierarchicalRelationRouter(
                embed_dim=task_hidden_dim,
                hierarchy=hierarchy,
                num_layers=int(relation_config.get("num_layers", 2)),
                num_heads=int(relation_config.get("num_heads", 4)),
                dropout=float(relation_config.get("dropout", 0.1)),
                ffn_mult=float(relation_config.get("ffn_mult", 2.0)),
                use_branch_embeddings=bool(relation_config.get("use_branch_embeddings", True)),
                residual_scale_init=float(relation_config.get("residual_scale_init", 0.0)),
            )
            if self.enable_hierarchical_relation_learning
            else None
        )
        self.task_relation_membership = {
            task_name: list(hierarchy.get("task_to_branches", {}).get(task_name, []))
            for task_name in self.task_names
        }
        self.task_heads = nn.ModuleDict()
        for task_spec in task_specs:
            output_dim = len(task_spec.output_classes) - 1 if task_spec.loss_type == "ordinal" else len(task_spec.output_classes)
            self.task_heads[task_spec.name] = nn.Linear(task_hidden_dim, output_dim)

    def _build_vit_backbone(self, in_channels: int, model_config: dict):
        hidden_size = int(model_config.get("hidden_size", 768))
        patch_size = int(model_config.get("patch_size", 32))
        image_size = int(model_config.get("image_size", 224))
        config = ViTConfig(
            hidden_size=hidden_size,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=3,
            num_hidden_layers=int(model_config.get("num_hidden_layers", 12)),
            num_attention_heads=int(model_config.get("num_attention_heads", 12)),
            intermediate_size=int(model_config.get("intermediate_size", 3072)),
            hidden_dropout_prob=float(model_config.get("hidden_dropout_prob", 0.0)),
            attention_probs_dropout_prob=float(model_config.get("attention_probs_dropout_prob", 0.0)),
            qkv_bias=True,
        )
        pretrained_source = str(model_config.get("pretrained_source", "") or "").strip()
        if pretrained_source:
            if Path(pretrained_source).exists():
                model = ViTModel(config)
                checkpoint = torch.load(pretrained_source, map_location="cpu")
                state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
                adapted = {}
                model_state = model.state_dict()
                for name, param in state_dict.items():
                    adapted_name = str(name).replace("module.", "").replace("vit.", "")
                    if adapted_name in model_state and tuple(model_state[adapted_name].shape) == tuple(param.shape):
                        adapted[adapted_name] = param
                model.load_state_dict(adapted, strict=False)
            else:
                model = ViTModel.from_pretrained(pretrained_source, ignore_mismatched_sizes=True)
        else:
            model = ViTModel(config)
        if in_channels != 3:
            self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
        return model

    def _extract_shared_hidden(self, image: torch.Tensor):
        if self.use_vit:
            if self.channel_adapter is not None:
                image = self.channel_adapter(image)
            hidden = self.embeddings(image)
            for layer_module in self.shared_encoder_layers:
                hidden = layer_module(hidden)[0]
            return hidden
        image_feat = self.backbone(image)
        return image_feat.flatten(1) if image_feat.ndim > 2 else image_feat

    def _extract_task_image_features(self, shared_hidden, task_name: str):
        if self.use_vit:
            hidden = shared_hidden
            for layer_module in self.task_encoder_layers[task_name]:
                hidden = layer_module(hidden)[0]
            hidden = self.task_layernorm[task_name](hidden)
            pooler = self.task_pooler[task_name]
            return hidden[:, 0] if isinstance(pooler, nn.Identity) else pooler(hidden)
        return shared_hidden

    def extract_feature_dict(self, image, meta):
        shared_hidden = self._extract_shared_hidden(image)
        shared_meta = self.meta_head(meta) if self.meta_head is not None else None

        task_outputs = {}
        base_task_features = {}
        for task_name in self.task_names:
            image_feat = self._extract_task_image_features(shared_hidden, task_name)
            if shared_meta is not None:
                fused_features = torch.cat([image_feat, shared_meta], dim=1)
                meta_feat = shared_meta
            else:
                fused_features = image_feat
                meta_feat = torch.zeros((image_feat.shape[0], 0), dtype=image_feat.dtype, device=image_feat.device)
            task_outputs[task_name] = {
                "image_features": image_feat,
                "meta_features": meta_feat,
                "fused_features": fused_features,
            }
            base_task_features[task_name] = self.task_necks[task_name](fused_features)

        if self.enable_hierarchical_relation_learning and self.hierarchy_router is not None:
            refined_task_features = self.hierarchy_router(base_task_features)
        else:
            refined_task_features = base_task_features

        for task_name in self.task_names:
            base_neck_features = base_task_features[task_name]
            neck_features = refined_task_features[task_name]
            task_outputs[task_name]["base_neck_features"] = base_neck_features
            task_outputs[task_name]["neck_features"] = neck_features
            task_outputs[task_name]["relation_delta"] = neck_features - base_neck_features
            task_outputs[task_name]["hierarchy_membership"] = self.task_relation_membership.get(task_name, [])
            task_outputs[task_name]["logits"] = self.task_heads[task_name](neck_features)
        return task_outputs

    def forward(self, image, meta):
        return {task_name: outputs["logits"] for task_name, outputs in self.extract_feature_dict(image, meta).items()}


def unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model
