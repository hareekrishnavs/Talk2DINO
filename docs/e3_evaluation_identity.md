# Canonical E3 evaluation identity

The authoritative machine-readable specification is
`evaluation_identities/e3_paired_soft_routing.toml`. The standalone verifier
does not import MMCV, Torch, model code, or dataset code during preflight.
It reads the DINOText constructor and the installed MMSegmentation
`COCOStuffDataset` source with Python's AST module, so omitted model flags and
the 171-class/no-background contract are checked without constructing either
the model or dataset.

## Static preflight

From the repository root, configuration-only validation is:

```bash
python verify_e3_identity.py preflight
```

The optional filesystem checks use explicit local paths and never download:

```bash
python verify_e3_identity.py preflight \
  --check-checkpoint \
  --weight-dir "$TALK2DINO_WEIGHT_DIR" \
  --dataset-root "$TALK2DINO_DATASET_ROOT"
```

`--check-checkpoint` checks the projection checkpoint at the repository path
`weights/vitb_mlp_infonce_paired_soft_routing_tau010.pth`. `--weight-dir`
checks `dinov2_vitb14_reg4_pretrain.pth` and `ViT-B-16.pt` without loading
them. `--dataset-root` checks the two validation directories without creating
a dataset. Static preflight always requires projection checkpoint loading to
be enabled, rejects dataset class/palette/metadata overrides (including inside
dataset wrappers), and verifies that the installed canonical dataset class has
171 classes whose first entry is not `background`. Dataset files are required
only when `--dataset-root` is supplied.

## Canonical cluster evaluation command

The evaluator's dataset configuration uses the repository-relative path
`data/coco_stuff164k`. Before evaluation, that path must be a directory or a
symlink to `$TALK2DINO_DATASET_ROOT`; do not replace a different existing
path. Set all four external locations to unique, writable paths:

```bash
export TALK2DINO_WEIGHT_DIR="/path/to/local/weights"
export TALK2DINO_DATASET_ROOT="/path/to/coco_stuff164k"
export TALK2DINO_OUTPUT_DIR="/path/to/unique/e3-output"
export TALK2DINO_LOG_DIR="/path/to/unique/e3-logs"

test -d "$TALK2DINO_DATASET_ROOT/images/val2017"
test -d "$TALK2DINO_DATASET_ROOT/annotations/val2017"
test -f "$TALK2DINO_WEIGHT_DIR/dinov2_vitb14_reg4_pretrain.pth"
test -f "$TALK2DINO_WEIGHT_DIR/ViT-B-16.pt"
test -f weights/vitb_mlp_infonce_paired_soft_routing_tau010.pth

if test ! -e data/coco_stuff164k; then
  mkdir -p data
  ln -s "$TALK2DINO_DATASET_ROOT" data/coco_stuff164k
fi
test "$(readlink -f data/coco_stuff164k)" = "$(readlink -f "$TALK2DINO_DATASET_ROOT")"

mkdir -p "$TALK2DINO_OUTPUT_DIR" "$TALK2DINO_LOG_DIR"

set -o pipefail
PYTHONPATH="$PWD" \
python -m torch.distributed.run --nproc_per_node=1 \
  src/open_vocabulary_segmentation/main.py \
  --eval \
  --eval_cfg src/open_vocabulary_segmentation/configs/stuff/dinotext_stuff_vitb_mlp_infonce_paired_soft_routing_tau010.yml \
  --eval_base_cfg src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml \
  --output "$TALK2DINO_OUTPUT_DIR" \
  2>&1 | tee "$TALK2DINO_LOG_DIR/e3-evaluation.log"
```

This is the repository's real single-GPU CLI. The selected configurations fix
PAMR off, use one augmentation, and contain no RWR, diffusion, graph-repair,
or COVER-DR option. Preflight also locks the effective
`sub_imagenet_template` prompt, requires the E3 projection checkpoint to be
loaded, and resolves all text/image token-selection settings against the real
DINOText constructor defaults.

## Result verification

Verify the real MMCV summary table in the captured log:

```bash
python verify_e3_identity.py verify-result \
  --log "$TALK2DINO_LOG_DIR/e3-evaluation.log"
```

A full-precision structured input uses percentage units:

```json
{
  "evaluated_images": 5000,
  "aAcc": 46.614213,
  "mIoU": 28.480169,
  "mAcc": 52.077968
}
```

```bash
python verify_e3_identity.py verify-result --metrics-json /path/to/metrics.json
```

The structured tolerance is the strict `0.000001` value defined by the identity
specification. Log tolerance is one half-unit in each value's last printed
decimal place. For two-decimal log tokens only, a `0.001` reproducibility
allowance is added to the `0.005` rounding uncertainty, giving a total
tolerance of `0.006`. Logs must print at least two decimal places, and the
additional allowance does not apply to higher-precision logs or structured
results.
