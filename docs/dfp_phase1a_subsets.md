# DFP Phase 1A — Step 2: deterministic evaluation subsets

Freezes exactly two immutable 500-image evaluation-subset manifests --
`coco_stuff_500` and `ade20k_500` -- selected from the authoritative
canonical validation splits already frozen by Step 1
(`docs/dfp_phase1a_protocol.md`). This step selects images; it does not
execute Phase 1A, generate predictions, or run propagation. See
`docs/dfp_phase1a_protocol.md` for the Phase 1A/1S/1B separation.

## Selection algorithm

`dfp_phase1a_deterministic_iterative_stratification v1`, implemented in
`src/dfp_phase1a_subset_manifest.py`. A self-contained, fully
deterministic two-fold specialization of the iterative-stratification
procedure (Sechidis, Tsoumakas & Vlahavas, "On the Stratification of
Multi-Label Data", ECML/PKDD 2011). See the module docstring for the
exact step-by-step specification; summarized:

1. For each class `c`, `class_totals[c]` = number of candidate images
   with at least one evaluated non-ignore pixel of class `c`.
2. `desired_selected[c]` is the nearest integer to
   `class_totals[c] * 500 / source_count`, computed with exact
   `fractions.Fraction` arithmetic. `desired_remainder[c] =
   class_totals[c] - desired_selected[c]`.
3. Repeatedly: the class with the smallest positive "remaining supply"
   (unassigned candidates still carrying it) is processed next; each of
   its still-unassigned candidates, in ascending image tie-break order,
   is assigned to a fold. A **hard capacity gate is checked first**: if
   only one fold still has remaining capacity, it is chosen
   unconditionally -- this guarantees the exact frozen fold sizes for
   genuinely multilabel data, since per-class desired counts are rounded
   independently and are not constrained to sum to 500 in aggregate.
   Only when both folds still have capacity does the label-demand
   comparison (`argmax` of remaining desired count for the triggering
   class) apply, with further ties broken by overall remaining fold
   target, then by image tie-break parity.
4. Label-empty candidates (an all-ignore mask) are assigned last, same
   capacity-gated rule.
5. The 500 selected IDs are reordered into the authoritative canonical
   validation order before being written to the manifest -- **selection
   order is never evaluation order**.

Ground truth class presence is used **only** for this preregistered
deterministic stratification (and for the underlying source-split
validation each dataset's own frozen identity already requires) — never
for prediction construction, and no model, feature extractor, or
checkpoint is touched anywhere in this step.

## Tie-breaking

`tie_break_namespace = "talk2dino-dfp-phase1a-subset-tie-break-v1"`.
Every tie-break integer is
`SHA256(namespace + NUL + dataset_key + NUL + canonical_key)`,
UTF-8-encoded, NUL-joined, interpreted as `int(hexdigest, 16)`. For an
image-level tie, `canonical_key` is the image's own canonical ID. For a
label-level tie (label-processing order, and the exact-one-half
rounding boundary), `canonical_key` is `f"class_index={c:04d}"` — a
format that cannot collide with a real COCO-Stuff (all-digit) or ADE20K
(`ADE_val_########`) image ID. No `random`, no NumPy/PyTorch RNG, no
dependence on set/dict iteration order, filesystem traversal order,
timestamps, hostnames, or process IDs anywhere in the algorithm.

## Path independence

The algorithm depends only on the **set** of candidate IDs and their
presence vectors, never on input traversal order: feeding it a reversed
or otherwise reordered candidate list produces the identical selected
set, and the mandatory final reordering step (to the true canonical
order) then produces the identical output order regardless. Generating
a manifest under two different, fresh temporary directories produces
byte-identical JSON.

## Manifest fields and digests

Each manifest (`evaluation_subsets/dfp_phase1a_coco_stuff_500.json`,
`evaluation_subsets/dfp_phase1a_ade20k_500.json`) records: schema/schema
version, dataset key/split/run mode, the protocol identity's relative
path and SHA-256, the dataset's source identity's relative path and
SHA-256, a `source_manifest_binding` (logical label + SHA-256 — COCO-
Stuff binds logically to `e12_matched_k11_k12_t320.toml`'s own
`[dataset]` section since no separate COCO-Stuff source-identity file
exists in this repository; ADE20K binds to an externally-verified
`talk2dino-ade20k-dataset-manifest-v1` source manifest), source/selected/
remainder counts, class count and class-names digest, the ignore policy
identifier, the frozen algorithm name/version and tie-break namespace/
encoding, the exact ordered selected IDs and their digest, a digest of
the full candidate universe, a digest of the full class-presence input,
full-source/selected/remainder per-class presence counts, the exact
desired per-class allocation, and a selection-trace digest that
self-checks the three count vectors' internal consistency. No
timestamps, hostnames, usernames, physical data roots, or absolute
paths are recorded — only a logical source-root label (e.g.
`"coco_stuff164k"`, `"ADEChallengeData2016"`), never a path.

## Immutable-before-prediction requirement

Per the frozen `[subset_selection_policy]` in the Step-1 protocol
identity, both manifests must be immutably committed **before** any
Phase 1A prediction is produced. **Step 3 begins only after both
manifests pass independent verification (`prepare_dfp_phase1a_subsets.py
verify` and `verify-source-binding`) and are committed to the
repository.**
