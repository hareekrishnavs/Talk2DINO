# COCO-Object val2017 mask materialization

## Why this stage exists

The repository already contains a complete, working COCO-Object protocol
— dataset class (`COCOObjectDataset`), dataset config (`coco.py`), and
canonical converter (`convert_dataset/convert_coco_object.py`) — but it
was never reachable: the `*_instanceTrainIds.png` label masks the dataset
class requires do not exist on disk under the configured data root. This
stage produces exactly those 5000 validation-split masks so a later
evaluation stage can consume the existing protocol without inventing a
new one.

**COCO-Object is a protocol on the same COCO images/annotations already
used for COCO-Stuff, not an independent dataset.** This stage does not
download, convert, or introduce any new imagery — it only derives a
different per-pixel label space from data already present.

## Canonical source files (not `instances_val2017.json`)

The task that motivated this stage assumed the converter's raw source was
`instances_val2017.json` (the COCO instance/panoptic annotation file).
Inspecting the actual, already-committed converter showed this is
**not** the case:

- `convert_dataset/convert_coco_object.py` globs
  `annotations/{train,val}2017/*.png`, excluding files with `TrainIds` in
  their name — i.e. it reads the raw, non-suffixed per-pixel COCO-Stuff
  category-ID masks (`<image_id>.png`) that already sit alongside the
  `<image_id>_labelTrainIds.png` files used for COCO-Stuff evaluation.
- There is **no polygon/RLE rasterization step** in this conversion, and
  no per-instance/crowd/overlap-order logic: the source is already a
  flattened, single-channel semantic label map, one label per pixel.
  Concepts like "crowd handling" or "overlapping-instance precedence"
  that apply to instance-JSON-based mask generation do not apply here.
- The instances JSON (`annotations_og/instances_val2017.json`) is
  unrelated to this specific converter and is not read by this stage.

This is a deliberate, evidence-based divergence from the task's assumed
input, made because the instructions explicitly required inspecting and
preserving the *actual* canonical converter's behavior rather than
redesigning it to match an assumption. `materialize_coco_object_val.py`
therefore takes `--source-masks` (the raw per-pixel mask directory) and
`--source-images` (the canonical `.jpg` directory), not a JSON path.

## Exact mapping and background semantics

The mapping table, `clsID_to_trID`, is **imported directly** from
`convert_dataset/convert_coco_object.py` (hash-checked against
`evaluation_identities/e12_coco_object_val_materialization.toml` before
use) — never re-derived or hand-copied. Its documented behavior:

- Raw category ID `k > 90` (every COCO-Stuff "stuff"-only category) maps
  to output class `0` ("background").
- Raw value `255` also maps to `0` — there is **no separate "ignore"
  output value** in this converter's design; the source's crowd/unlabeled
  convention is folded into background, not preserved as
  `ignore_index=255` in the *output* mask.
- Every other raw category ID `k <= 90` maps to a fixed output class in
  `1..80`, matching `COCOObjectDataset.CLASSES` (`'background'` + 80
  thing categories).
- Empirically confirmed on the real val2017 data: **every raw pixel
  value present is covered by the table** (0 uncovered values across all
  5000 raw masks). `apply_canonical_mapping` still fails closed on any
  uncovered value as a safety net — the original script would silently
  leave such a pixel unchanged, which this materializer never does.
- Cross-validated: 100 real val2017 images (first, last, 98 random)
  produce byte-identical output between the extracted LUT and the real
  `convert_to_trainID` function.

## Val-only scope

`materialize_coco_object_val.py` only accepts `--source-masks`/
`--source-images` directories whose final path component is exactly
`val2017` (the identity's `protocol.split`); it has no flag capable of
selecting `train2017` or any other split name. `--test-limit` (bounded,
explicitly noncanonical) requires `--allow-noncanonical-test-mode` and
never writes a completion manifest — it exists only for CPU-safe testing
under `/tmp`.

## Output-root isolation

`--output-root` is rejected if it resolves inside, or overlaps, the
source COCO-Stuff annotation root (`coco_stuff164k/annotations`, recorded
as `policy.source_annotation_root_marker` in the identity). The derived
root's `images/val2017` is a validated symlink to the canonical source
images directory — no image bytes are ever copied.

## Storage / CPU estimate

5000 single-channel, small-dimension (COCO val2017 average ~640×480)
uint8 PNG masks: well under 1 GB total (COCO-Stuff's own equivalent
`_labelTrainIds.png` set for the same 5000 images is a useful reference
point and is already present at a similar order of magnitude on the same
filesystem). The conversion is a per-image vectorized NumPy LUT lookup —
no model, no GPU; a full 5000-image run is expected to complete in low
single-digit minutes of CPU time on a single core, comfortably within the
SBATCH script's conservative `--time=01:00:00` / `--cpus-per-task=4`
/ `--mem=8G` allocation.

## Resumability

Every image updates the checkpoint immediately after its mask is
validated and atomically installed (temp-file write, reopen-and-compare,
`os.replace`). On `--resume`, the checkpoint's structure, canonical-order
binding, and **every already-completed mask's decoded/encoded hash** are
re-verified against the files actually on disk before continuing — a
corrupted, tampered, or missing "completed" mask fails the resume closed
rather than being silently trusted.

## Manifest / provenance

The final manifest (`talk2dino-coco-object-val-materialization-manifest-v1`)
is written only once all images validate, and records: the canonical
image-order digest, a `source_masks_digest` (a stand-in for a single
"source JSON hash" — this converter has none — computed over every
source raw mask's own content hash), aggregate decoded/encoded-file
digests, the full 81-class pixel histogram, foreground/all-background
mask counts, and the exact converter/dataset-class/dataset-config SHA256s
it was produced under. `complete`/`final` are both required `true`; an
incomplete run can never produce a document `verify_record` accepts.

## Verification

`verify_coco_object_val_materialization.py` never trusts the manifest's
own claims for `verify-output`: it independently re-scans all installed
masks, recomputes every hash/digest/histogram from scratch, and cross-
checks against source image dimensions read fresh from disk. A fixed-
selection spot-check (first canonical image, last, an all-background
image if one exists, the image with the most distinct foreground
classes, the image with the smallest nonzero-class pixel footprint, plus
enough evenly-spaced additional images to reach at least 20) re-applies
the hash-verified mapping table through code that does **not** call the
production `apply_canonical_mapping`/`convert_one_image` — a
per-raw-value boolean-mask loop instead of the vectorized LUT — and
requires pixel-exact equality against the installed mask.

Note on Section-10-style selection criteria that do not apply to this
converter: "crowded image" and "image with overlapping instances" are
instance-level concepts that have no meaning for this semantic
(category-level) conversion; the spot-check substitutes "most distinct
foreground classes represented" and "smallest nonzero foreground-class
pixel footprint" as the closest applicable analogues, documented here
rather than silently omitted.

## Failure recovery

Every expected failure (`CocoObjectValMaterializationError`,
`CocoObjectValMaterializationIdentityError`, `OSError`, `ValueError`)
exits `2` with a concise `COCO-OBJECT VAL MATERIALIZATION FAIL: ...`
message and no traceback; `KeyboardInterrupt`/`SystemExit` are never
caught by either CLI's top-level handler. A failed run's checkpoint and
any already-installed masks are left in place for `--resume`, never
deleted.

## Downstream usage

Once materialized, `--output-root` is a directly usable data root for
the existing, unmodified `COCOObjectDataset`/`coco.py` dataset config
(`images/val2017`, `annotations/val2017` in the exact structure
`data.test` already expects) — the later COCO-Object evaluation stage
will pass this derived root explicitly; this stage does not modify the
committed dataset config.

## Reproduction commands (not run during implementation)

```
module load opencv/4.14.0
source /scratch/haree/venv/talk2dino-a100/bin/activate

python verify_coco_object_val_materialization.py preflight \
    --repo-root . \
    --source-masks /scratch/haree/coco_stuff164k/annotations/val2017 \
    --source-images /scratch/haree/coco_stuff164k/images/val2017 \
    --output-root /scratch/haree/coco_object_protocol

python materialize_coco_object_val.py \
    --repo-root . \
    --source-masks /scratch/haree/coco_stuff164k/annotations/val2017 \
    --source-images /scratch/haree/coco_stuff164k/images/val2017 \
    --output-root /scratch/haree/coco_object_protocol \
    --checkpoint /scratch/haree/coco_object_protocol/checkpoints/checkpoint-manual.json \
    --manifest /scratch/haree/coco_object_protocol/manifests/manifest-manual.json

python verify_coco_object_val_materialization.py verify-output \
    --repo-root . \
    --manifest /scratch/haree/coco_object_protocol/manifests/manifest-manual.json \
    --output-root /scratch/haree/coco_object_protocol \
    --source-masks /scratch/haree/coco_stuff164k/annotations/val2017 \
    --source-images /scratch/haree/coco_stuff164k/images/val2017
```

Or, on the cluster: `sbatch scripts/slurm/e12_materialize_coco_object_val_cpu.sbatch`
(not submitted during implementation).
