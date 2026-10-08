# Recursive RF-DETR Medium

`RFDETRMedium` uses three refinement stages by default. Each stage reuses the same decoder, detection heads, and query
slots. The backbone runs once, and encoder proposals initialize only the first stage. Later stages compute memory notes
with `notes = MLP(concat(previous_content, previous_notes))`, use `previous_content + notes` as queries, and carry the
previous predicted boxes as spatial references. Notes reset for each image batch; gradients flow across stages.

Install the tracked source:

```bash
python -m pip install -e './rf-detr[train]'
```

Verify locally before pushing:

```bash
python rf-detr/scripts/smoke_recursive_decoder.py
python rf-detr/scripts/smoke_recursive_decoder.py --group-detr 13
cd rf-detr
python -m pytest tests/models/test_recursive_decoder.py
```

The smoke test adapts the supplied test harness to this version's config-native builders. It creates a real
`RFDETRMedium(pretrain_weights=None)` on CPU with `[2, 3, 128, 128]` images, 50 queries per group, 80 classes, and dummy COCO
targets. It checks finite outputs and gradients, Hungarian losses for every stage, and two AdamW updates under anomaly
detection. It does not download pretrained weights.

Train with the Medium wrapper:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(
    recursive_stages=3,
    recursive_stage_weights=[1.0, 1.0, 1.0],
)
model.train(
    dataset_dir="/path/to/coco",
    output_dir="runs_recursive",
    epochs=12,
    batch_size=4,
    lr=1e-4,
)
```

`recursive_outputs` contains every stage's final predictions and decoder-layer auxiliaries. The top-level `pred_logits`
and `pred_boxes` are the final stage. The criterion makes independent Hungarian assignments for every stage and auxiliary
layer. The training sum applies each stage's weight exactly once; the existing encoder auxiliary loss is added once.
Earlier stages use loss names such as `loss_bbox_recursive_0` and `loss_bbox_0_recursive_0`; final-stage loss names remain
`loss_bbox` and `loss_bbox_0` for compatibility with the training loop. Omit `recursive_stage_weights` for unit weights.
Use `recursive_stages=1` for the standard single-pass Medium decoder. Recursive segmentation is not supported.
The existing `train_baseline*.py` scripts explicitly use this single-pass setting.

The dataset package and its tests were restored from the repository's pinned upstream revision `6e1620e` because the
source conversion omitted them under an overly broad `datasets/` ignore rule.
