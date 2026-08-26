# VOC2012 dataset-source contract (V20/V21)

## Scope

This is a **dataset-source contract**, not an evaluator. It establishes
a fail-closed, reproducible identity/manifest for the PASCAL VOC 2012
segmentation validation split shared by Talk2DINO's V20 (no-background)
and V21 (with-background) protocols. It does **not** implement a
GPU evaluator, extract DINO/CLIP features, load a model, or initialize
CUDA. The existing COCO-2017-trained Talk2DINO bridge (projection head)
remains frozen and out of scope here.

## Upstream source

- Dataset: PASCAL VOC 2012 (`http://host.robots.ox.ac.uk/pascal/VOC/voc2012/`)
- Archive top-level directory: `VOCdevkit`
- Extracted root used by this contract: `$VOCDEVKIT_ROOT/VOCdevkit/VOC2012`
  (or `$VOCDEVKIT_ROOT` itself if it already *is* the `VOC2012` directory
  -- root resolution is fail-closed and detects both, reporting ambiguity
  if more than one candidate satisfies the root contract; it never
  assumes a fixed nesting depth).

Discovered real layout on this cluster (do not embed this in tracked
scripts -- use `$VOCDEVKIT_ROOT` as a placeholder for whatever the real
path is on a given machine):

```
$VOCDEVKIT_ROOT/VOCdevkit/VOC2012/
  JPEGImages/                          # 17,125 JPEGs (all splits)
  SegmentationClass/                   # 2,913 PNGs (train+val)
  SegmentationObject/
  Annotations/
  ImageSets/Segmentation/
    train.txt        # 1,464 ids
    val.txt           # 1,449 ids  <- the split this contract governs
    trainval.txt      # 2,913 ids
```

## Was conversion required? No.

The repository's own `PascalVOCDataset20`/`PascalVOCDataset` classes
consume `SegmentationClass/*.png` directly (`img_suffix='.jpg'`,
`seg_map_suffix='.png'`), and a full scan of all 1,449 validation masks
confirmed every one decodes as a single-channel palettized (`'P'` mode)
PNG whose only pixel values are `{0..20, 255}` -- exactly the legal V21
domain (21 classes + ignore). **No mask conversion, no VOCaug
installation, and no duplicate mask generation is needed or was
performed.** Validation against the archive as downloaded and extracted
*is* the preparation step for this contract.

## V20 vs V21: same files, different runtime label transform

Both protocols read the identical `JPEGImages/*.jpg` +
`SegmentationClass/*.png` files, the identical `ImageSets/Segmentation/val.txt`
split (order authority: file line order, each line stripped -- exactly
`mmseg.datasets.custom.CustomDataset.load_annotations`'s own
split-driven ordering, never re-sorted, never derived from a directory
listing). The only difference is which dataset class interprets the raw
pixel values:

| | V21 (with background) | V20 (no background) |
|---|---|---|
| Dataset class | `mmseg.datasets.PascalVOCDataset` (stock, unmodified) | `PascalVOCDataset20` (repo-local, `src/open_vocabulary_segmentation/segmentation/datasets/pascal_voc.py`) |
| Class count | 21 | 20 |
| `CLASSES[0]` | `'background'` | *(no background entry)* |
| `reduce_zero_label` | `False` (mmseg default) | `True` |
| Label transform | none -- raw pixel values 0..20 used directly as the 21 evaluated classes | raw 0 (background) &rarr; 255 (ignore); raw 1..20 &rarr; 0..19; raw 255 (void) stays 255 |
| `ignore_index` | 255 | 255 |

The V20 transform is `mmseg`'s own, standard `reduce_zero_label`
mechanism (`mmseg.datasets.pipelines.LoadAnnotations.__call__`), applied
because `PascalVOCDataset20.__init__` passes `reduce_zero_label=True` to
`CustomDataset.__init__` -- never a redesigned or independently-derived
mapping. `V21_CLASSES == ('background',) + V20_CLASSES`, same order.

Both `_base_` dataset configs (`pascal_voc12.py` for V21,
`pascal_voc12_20.py` for V20) additionally agree on: `data_root =
"./data/VOCdevkit/VOC2012"`, `img_dir = "JPEGImages"`, `ann_dir =
"SegmentationClass"`, `split = "ImageSets/Segmentation/val.txt"`,
and the sliding-window evaluation geometry `mode="slide"`,
`crop_size=(448, 448)`, `stride=(224, 224)`.

## VOCaug

Not required. This contract governs **validation-only** evaluation
against the standard `SegmentationClass` masks; the `train.txt`/
`trainval.txt` splits and any augmented (SBD/VOCaug) training masks are
outside its scope entirely. No authoritative config in this repository
requires VOCaug for the V20/V21 evaluation pipelines this contract
feeds.

## No bridge training or feature extraction

This stage performs dataset-source verification only. No DINO/CLIP
feature extraction, no model loading, and no CUDA initialization occur
under `evaluation_identities/e12_voc2012_dataset_source.toml` or its
supporting modules/CLI. The existing COCO-2017-trained Talk2DINO
projection head remains frozen for the development track; nothing here
trains or extracts features for a VOC2012-specific bridge.

## Files

- `evaluation_identities/e12_voc2012_dataset_source.toml` -- sole
  authority for every scientific value described above.
- `src/voc2012_dataset_identity.py` -- strict TOML loading/validation,
  plus live relational checks (V20 class list vs. the hash-pinned
  source file; V21 class list vs. the installed `mmsegmentation`
  package).
- `src/voc2012_dataset_manifest.py` -- fail-closed dataset-root
  resolution, canonical validation-id ordering, exhaustive per-file
  scan (decode, dimension, label-domain, hash), and deterministic
  manifest build/verify.
- `verify_voc2012_dataset.py` -- CLI (`preflight` /
  `generate-manifest` / `verify-manifest`).

## Commands

Set `TALK2DINO_VENV` to the path of your project virtual environment
before running any of these commands (never a hardcoded, machine-specific
path).

```sh
module load opencv/4.14.0
source "$TALK2DINO_VENV/bin/activate"

python verify_voc2012_dataset.py preflight \
    --repo-root . \
    --data-root $VOCDEVKIT_ROOT/VOCdevkit

python verify_voc2012_dataset.py generate-manifest \
    --repo-root . \
    --data-root $VOCDEVKIT_ROOT/VOCdevkit \
    --output /scratch/$USER/e12_voc2012/voc2012_manifest.json

python verify_voc2012_dataset.py verify-manifest \
    --repo-root . \
    --data-root $VOCDEVKIT_ROOT/VOCdevkit \
    --manifest /scratch/$USER/e12_voc2012/voc2012_manifest.json
```

`--data-root` accepts either the `VOCdevkit` directory or the `VOC2012`
directory directly; resolution is fail-closed and reports ambiguity
rather than guessing if more than one candidate qualifies.

## Expected values

- Validation image count: **1,449** (verified against the real archive).
- Legal raw mask label domain: `{0, 1, ..., 20, 255}` (21 classes +
  ignore); any other value is rejected.
- `ignore_index = 255` for both protocols.
- The manifest is path-independent: it records `source_root_logical_label
  = "VOC2012"` only, never the real filesystem path the data was read
  from.

## Manifest contents

Deterministic (byte-identical across runs against unchanged data except
`generated_at_utc`): schema/identity name+hash, split, image count,
V20/V21 class counts, `split_file_sha256`, an ordered `image_order_digest`,
three ordered content digests (`image_content_digest`,
`mask_encoded_content_digest`, `mask_decoded_content_digest` -- each a
SHA-256 over the ordered per-file SHA-256 list, giving collision-resistant
evidence for every validation image and mask without embedding 1,449
raw hash entries), the observed label set, a 21-class pixel histogram,
ignore-pixel count, and a dimension-reconciliation summary.

## Next stage (not part of this task)

Implement the shared V20/V21 GPU evaluation adapter and pilot20/pilot100
runs, reusing this identity's dataset-loader bindings exactly as the
COCO-Object protocol confirmation stage reused its own materialization
identity -- never redeclaring the class list, label transform, or
crop/stride as new Python literals.
