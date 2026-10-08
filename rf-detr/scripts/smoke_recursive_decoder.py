# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Verify a real RFDETRMedium on CPU with synthetic COCO targets and AdamW."""

import argparse
import copy

import torch

from rfdetr import RFDETRMedium
from rfdetr.config import TrainConfig
from rfdetr.models import build_criterion_from_config


def run_sandbox_smoke_test(recursive_stages: int = 3, steps: int = 2, group_detr: int = 1) -> None:
    """Exercise forward, independent Hungarian assignments, backward, and optimizer updates.

    Args:
        recursive_stages: Number of shared decoder refinement passes.
        steps: Optimizer steps, each using a fresh computation graph.
        group_detr: Query groups used during training.
    """
    torch.manual_seed(42)
    torch.set_num_threads(2)
    detector = RFDETRMedium(  # type: ignore[no-untyped-call]  # Upstream wrapper constructor is untyped.
        device="cpu",
        pretrain_weights=None,
        resolution=128,
        num_queries=50,
        num_select=50,
        num_classes=80,
        group_detr=group_detr,
        recursive_stages=recursive_stages,
        amp=False,
        fused_optimizer=False,
    )
    model = detector.model.model.train()
    criterion, _ = build_criterion_from_config(detector.model_config, TrainConfig(dataset_dir="."))
    criterion.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dummy_images = torch.randn(2, 3, 128, 128)
    dummy_targets = [
        {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[0.7, 0.6, 0.3, 0.3]]), "labels": torch.tensor([2])},
    ]
    print(f"RFDETRMedium: CPU, images={tuple(dummy_images.shape)}, T={recursive_stages}", flush=True)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autograd.detect_anomaly():
            outputs = model(dummy_images)
            stages = outputs["recursive_outputs"]
            assert len(stages) == recursive_stages
            for stage in stages:
                assert stage["pred_boxes"].shape == (2, 50 * group_detr, 4)
                assert stage["pred_logits"].shape == (2, 50 * group_detr, 81)
                assert torch.isfinite(stage["pred_boxes"]).all()
                assert torch.isfinite(stage["pred_logits"]).all()
            loss_dict = criterion(outputs, dummy_targets)
            for stage_id in range(recursive_stages - 1):
                assert f"loss_ce_recursive_{stage_id}" in loss_dict
                assert f"loss_bbox_recursive_{stage_id}" in loss_dict
                assert f"loss_giou_recursive_{stage_id}" in loss_dict
            total_loss = sum(
                loss_dict[key] * criterion.weight_dict[key] for key in loss_dict if key in criterion.weight_dict
            )
            assert torch.isfinite(total_loss)
            total_loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), f"Nonfinite gradient: {name}"
        if recursive_stages > 1:
            for parameter in model.notes_mlp.parameters():
                assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        old_weight = model.class_embed.weight.detach().clone()
        optimizer.step()
        assert not torch.equal(old_weight, model.class_embed.weight)
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        print(
            f"Step {step + 1}: forward, Hungarian loss, backward, AdamW passed; loss={total_loss.item():.6f}",
            flush=True,
        )
    model.eval()
    criterion.eval()
    with torch.no_grad():
        evaluation = model(dummy_images)
        assert evaluation["pred_boxes"].shape == (2, 50, 4)
        assert all(torch.isfinite(loss).all() for loss in criterion(evaluation, dummy_targets).values())
        exported = copy.deepcopy(model)
        exported.export()
        export_boxes, export_logits = exported(dummy_images)
        torch.testing.assert_close(export_boxes, evaluation["pred_boxes"], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(export_logits, evaluation["pred_logits"], rtol=1e-4, atol=1e-5)
    print("Evaluation and export agree with the final refinement stage.", flush=True)
    print("SUCCESS: Sandbox verification passed with 0 errors.", flush=True)


def main() -> None:
    """Parse smoke-test options and run local verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recursive-stages", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--group-detr", type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    run_sandbox_smoke_test(args.recursive_stages, args.steps, args.group_detr)


if __name__ == "__main__":
    main()
