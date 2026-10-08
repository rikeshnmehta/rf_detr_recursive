# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for shared recursive decoding, memory notes, and stage losses."""

import copy
from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import nn

from rfdetr.config import RFDETRMediumConfig, TrainConfig
from rfdetr.models import MODEL_DEFAULTS, build_criterion_from_config
from rfdetr.models.lwdetr import LWDETR
from rfdetr.models.math import inverse_sigmoid
from rfdetr.models.transformer import Transformer
from rfdetr.utilities.tensors import NestedTensor


class TinyBackbone(nn.Module):
    """Small differentiable features for tests of the actual RF-DETR decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 32, kernel_size=1)
        self.calls = 0

    def forward(self, samples: NestedTensor) -> tuple[list[NestedTensor], list[torch.Tensor]]:
        """Project images into one feature level and count backbone evaluations."""
        self.calls += 1
        feature = self.projection(samples.tensors)
        return [NestedTensor(feature, samples.mask)], [torch.zeros_like(feature)]


def make_model(
    stages: int = 3, group_detr: int = 1, bbox_reparam: bool = True, lite_refpoint_refine: bool = True
) -> LWDETR:
    """Build the real decoder with small feature dimensions for fast regression tests."""
    torch.manual_seed(42)
    transformer = Transformer(
        d_model=32,
        sa_nhead=4,
        ca_nhead=4,
        num_queries=5,
        num_decoder_layers=2,
        dim_feedforward=64,
        return_intermediate_dec=True,
        group_detr=group_detr,
        two_stage=True,
        num_feature_levels=1,
        dec_n_points=2,
        lite_refpoint_refine=lite_refpoint_refine,
        bbox_reparam=bbox_reparam,
    )
    return LWDETR(
        TinyBackbone(),
        transformer,
        None,
        num_classes=4,
        num_queries=5,
        aux_loss=True,
        group_detr=group_detr,
        two_stage=True,
        lite_refpoint_refine=lite_refpoint_refine,
        bbox_reparam=bbox_reparam,
        recursive_stages=stages,
    )


def make_targets(empty: bool = False) -> list[dict[str, torch.Tensor]]:
    """Return a COCO-style batch with normalized cxcywh boxes."""
    if empty:
        return [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)} for _ in range(2)]
    return [
        {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[0.7, 0.6, 0.3, 0.3]]), "labels": torch.tensor([2])},
    ]


def make_criterion(stages: int = 3, weights: list[float] | None = None, aux_loss: bool = True) -> nn.Module:
    """Build the production criterion through RFDETRMediumConfig."""
    config = RFDETRMediumConfig(
        num_classes=3,
        dec_layers=2,
        group_detr=1,
        recursive_stages=stages,
        recursive_stage_weights=weights,
        pretrain_weights=None,
        device="cpu",
    )
    criterion, _ = build_criterion_from_config(
        config, TrainConfig(dataset_dir="."), defaults=replace(MODEL_DEFAULTS, aux_loss=aux_loss)
    )
    return criterion


class TestRecursiveDecoder:
    """Refinement must reuse query slots, proposals, and a differentiable stage graph."""

    @pytest.mark.parametrize("training", [pytest.param(True, id="train"), pytest.param(False, id="eval")])
    def test_shared_decoder_fixed_queries_and_one_backbone_call(self, training: bool) -> None:
        """All stages share weights and query counts; encoder proposals are generated once."""
        model = make_model(group_detr=3).train(training)
        decoded = []
        proposed = []
        model.transformer.decoder.register_forward_hook(lambda module, args, output: decoded.append(id(module)))
        for head in model.transformer.enc_output:
            head.register_forward_hook(lambda module, args, output: proposed.append(id(module)))
        outputs = model(torch.randn(2, 3, 4, 4))
        assert len(outputs["recursive_outputs"]) == 3
        assert model.backbone.calls == 1
        assert decoded == [id(model.transformer.decoder)] * 3
        assert len(proposed) == (3 if training else 1)
        for stage in outputs["recursive_outputs"]:
            assert stage["pred_boxes"].shape == (2, 15 if training else 5, 4)
            assert len(stage["aux_outputs"]) == 1
        assert outputs["pred_boxes"] is outputs["recursive_outputs"][-1]["pred_boxes"]
        assert all("enc_outputs" not in stage for stage in outputs["recursive_outputs"])

    @pytest.mark.parametrize("bbox_reparam", [pytest.param(True, id="reparam"), pytest.param(False, id="sigmoid")])
    def test_notes_and_references_carry_actual_previous_predictions(self, bbox_reparam: bool) -> None:
        """Notes consume previous content and notes; references come from previous boxes."""
        model = make_model(bbox_reparam=bbox_reparam)
        decoder_inputs = []
        decoder_outputs = []
        notes_inputs = []
        notes_outputs = []
        model.transformer.register_forward_pre_hook(lambda module, args: decoder_inputs.append(args))
        model.transformer.register_forward_hook(lambda module, args, output: decoder_outputs.append(output))
        model.notes_mlp.register_forward_pre_hook(lambda module, args: notes_inputs.append(args[0]))
        model.notes_mlp.register_forward_hook(lambda module, args, output: notes_outputs.append(output))
        outputs = model(torch.randn(2, 3, 4, 4))
        assert decoder_inputs[0][4].ndim == 2
        for index in range(2):
            content = decoder_outputs[index][0][-1]
            notes = torch.zeros_like(content) if index == 0 else notes_outputs[index - 1]
            torch.testing.assert_close(notes_inputs[index], torch.cat([content, notes], dim=-1))
            torch.testing.assert_close(decoder_inputs[index + 1][4], content + notes_outputs[index])
            boxes = outputs["recursive_outputs"][index]["pred_boxes"]
            expected_refs = boxes if bbox_reparam else inverse_sigmoid(boxes)
            torch.testing.assert_close(decoder_inputs[index + 1][3], expected_refs)

    @pytest.mark.parametrize("lite", [pytest.param(True, id="lite"), pytest.param(False, id="iterative")])
    @pytest.mark.parametrize("bbox_reparam", [pytest.param(True, id="reparam"), pytest.param(False, id="sigmoid")])
    def test_final_loss_backpropagates_through_earlier_boxes(self, lite: bool, bbox_reparam: bool) -> None:
        """Even final-stage-only supervision reaches earlier spatial predictions and notes."""
        model = make_model(lite_refpoint_refine=lite, bbox_reparam=bbox_reparam)
        outputs = model(torch.randn(2, 3, 4, 4))
        for stage in outputs["recursive_outputs"]:
            stage["pred_boxes"].retain_grad()
        final = outputs["recursive_outputs"][-1]
        losses = make_criterion(stages=1)(final, make_targets())
        (losses["loss_bbox"] + losses["loss_giou"] + losses["loss_ce"]).backward()
        for stage in outputs["recursive_outputs"][:-1]:
            gradient = stage["pred_boxes"].grad
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        for parameter in model.notes_mlp.parameters():
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0

    def test_single_stage_and_no_cross_batch_memory(self) -> None:
        """T=1 retains the standard path, and notes reset on every forward call."""
        model = make_model(stages=1).eval()
        images = torch.randn(2, 3, 4, 4)
        outputs = model(images)
        assert len(outputs["recursive_outputs"]) == 1
        assert model.notes_mlp is None
        model = make_model().eval()
        torch.testing.assert_close(model(images)["pred_boxes"], model(images)["pred_boxes"])

    def test_first_stage_matches_standard_medium_decoder(self) -> None:
        """With identical weights, stage 1 is exactly the standard single decoder pass."""
        recursive = make_model().eval()
        standard = copy.deepcopy(recursive)
        standard.recursive_stages = 1
        standard.notes_mlp = None
        images = torch.randn(2, 3, 4, 4)
        first = recursive(images)["recursive_outputs"][0]
        baseline = standard(images)
        torch.testing.assert_close(first["pred_boxes"], baseline["pred_boxes"])
        torch.testing.assert_close(first["pred_logits"], baseline["pred_logits"])

    def test_without_encoder_proposals(self) -> None:
        """Refinement also preserves query slots when two_stage=False."""
        model = make_model()
        model.two_stage = model.transformer.two_stage = False
        outputs = model(torch.randn(2, 3, 4, 4))
        assert len(outputs["recursive_outputs"]) == 3
        assert "enc_outputs" not in outputs
        assert all(stage["pred_boxes"].shape == (2, 5, 4) for stage in outputs["recursive_outputs"])


class TestRecursiveLoss:
    """Stage weights must reach the training sum with no duplicate final or encoder loss."""

    @pytest.mark.parametrize("aux_loss", [pytest.param(True, id="aux"), pytest.param(False, id="no-aux")])
    def test_weighted_total_equals_sum_of_independent_stage_losses(self, aux_loss: bool) -> None:
        """Check Hungarian call count and an independently assembled weighted loss."""
        model = make_model()
        model.aux_loss = aux_loss
        outputs = model(torch.randn(2, 3, 4, 4))
        targets = make_targets()
        lambdas = [0.25, 0.5, 1.5]
        criterion = make_criterion(weights=lambdas, aux_loss=aux_loss)
        matched = []
        criterion.matcher.register_forward_hook(lambda module, args, output: matched.append(args[0]))
        losses = criterion(outputs, targets)
        assert len(matched) == 3 * (2 if aux_loss else 1) + 1
        actual = sum(losses[key] * criterion.weight_dict[key] for key in losses if key in criterion.weight_dict)
        standard = make_criterion(stages=1, aux_loss=aux_loss)
        expected = torch.zeros_like(actual)
        for weight, stage in zip(lambdas, outputs["recursive_outputs"]):
            stage_losses = standard(stage, targets)
            expected = expected + weight * sum(
                stage_losses[key] * standard.weight_dict[key] for key in stage_losses if key in standard.weight_dict
            )
        if aux_loss:
            encoder_losses = standard(outputs["enc_outputs"], targets)
            expected = expected + sum(
                encoder_losses[key] * standard.weight_dict[key] for key in encoder_losses if key in standard.weight_dict
            )
        torch.testing.assert_close(actual, expected)

    def test_empty_targets_backward(self) -> None:
        """Empty COCO batches still produce finite losses and valid gradients."""
        outputs = make_model()(torch.randn(2, 3, 4, 4))
        criterion = make_criterion()
        losses = criterion(outputs, make_targets(empty=True))
        total = sum(losses[key] * criterion.weight_dict[key] for key in losses if key in criterion.weight_dict)
        assert torch.isfinite(total)
        total.backward()

    def test_rejects_mismatched_stage_count(self) -> None:
        """An incompatible model/criterion stage configuration must fail explicitly."""
        outputs = make_model()(torch.randn(2, 3, 4, 4))
        with pytest.raises(ValueError, match="recursive_outputs length"):
            make_criterion(stages=1)(outputs, make_targets())


class TestRecursiveConfig:
    """The Medium variant enables recursion and validates stage configuration."""

    def test_medium_defaults_to_three_stages(self) -> None:
        """Medium enables T_max=3 by default, while preserving its backbone configuration."""
        config = RFDETRMediumConfig()
        assert config.recursive_stages == 3
        assert config.dec_layers == 4 and config.patch_size == 16 and config.num_windows == 2

    @pytest.mark.parametrize("stages", [pytest.param(0, id="zero"), pytest.param(-1, id="negative")])
    def test_rejects_invalid_stage_count(self, stages: int) -> None:
        """At least one decoder stage is required."""
        with pytest.raises(ValueError, match="recursive_stages"):
            RFDETRMediumConfig(recursive_stages=stages)

    @pytest.mark.parametrize(
        "weights",
        [
            pytest.param([1.0], id="length"),
            pytest.param([-1.0, 1.0, 1.0], id="negative"),
            pytest.param([float("nan"), 1.0, 1.0], id="nan"),
            pytest.param([float("inf"), 1.0, 1.0], id="inf"),
            pytest.param([0.0, 0.0, 0.0], id="all-zero"),
        ],
    )
    def test_rejects_invalid_weights(self, weights: list[float]) -> None:
        """One finite nonnegative weight per stage is required, with a positive total."""
        with pytest.raises(ValueError, match="recursive_stage_weights"):
            RFDETRMediumConfig(recursive_stage_weights=weights)

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"dec_layers": 0}, id="no-decoder"),
            pytest.param({"segmentation_head": True}, id="segmentation"),
        ],
    )
    def test_rejects_unsupported_recursive_architectures(self, kwargs: dict[str, Any]) -> None:
        """Recursive refinement is a detection decoder stage."""
        with pytest.raises(ValueError, match="recursive"):
            RFDETRMediumConfig(**kwargs)
