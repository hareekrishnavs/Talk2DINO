"""D1, D3, D4, D5: COCO Captions loading, noun extraction/vocabulary, and
the train/val-by-image caption split. No masks, no pixel labels, no COCO
class list are read anywhere in this module -- captions and raw images
only."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

COCO_CAPTIONS_TRAIN = Path("/scratch/haree/coco_stuff164k/annotations_og/captions_train2017.json")
COCO_IMAGES_TRAIN = Path("/scratch/haree/coco_stuff164k/images/train2017")


def load_captions(captions_json: Path = COCO_CAPTIONS_TRAIN) -> dict[int, dict[str, Any]]:
    """Returns {image_id: {"file_name": str, "captions": [str, ...]}}."""
    data = json.loads(Path(captions_json).read_text())
    images = {row["id"]: {"file_name": row["file_name"], "captions": []} for row in data["images"]}
    for ann in data["annotations"]:
        images[ann["image_id"]]["captions"].append(ann["caption"])
    return images


_STOP_NOUNS = {"image", "photo", "picture", "photograph", "view"}  # generic, near-universal, low signal


def extract_nouns(caption: str, nlp) -> list[str]:
    """Lowercased noun lemmas from a caption via the given spaCy pipeline.
    Excludes pronouns and a small stoplist of near-universal, low-signal
    nouns that appear in almost every caption regardless of image content."""
    doc = nlp(caption)
    nouns = []
    for token in doc:
        if token.pos_ != "NOUN":
            continue
        lemma = token.lemma_.lower().strip()
        if not lemma.isalpha() or lemma in _STOP_NOUNS:
            continue
        nouns.append(lemma)
    return nouns


def build_noun_vocabulary(
    images: dict[int, dict[str, Any]], nlp, *, min_count: int = 5,
) -> tuple[list[str], dict[int, list[str]]]:
    """Extracts nouns from every caption, keeps nouns occurring at least
    `min_count` times across the corpus. Returns (sorted vocabulary,
    {image_id: [present nouns, deduplicated, vocabulary-filtered]})."""
    from collections import Counter

    counts: Counter[str] = Counter()
    per_image_nouns: dict[int, list[str]] = {}
    for image_id, entry in images.items():
        nouns_this_image: set[str] = set()
        for caption in entry["captions"]:
            nouns_this_image.update(extract_nouns(caption, nlp))
        per_image_nouns[image_id] = sorted(nouns_this_image)
        counts.update(nouns_this_image)

    vocabulary = sorted(noun for noun, count in counts.items() if count >= min_count)
    vocab_set = set(vocabulary)
    per_image_nouns = {
        image_id: [n for n in nouns if n in vocab_set]
        for image_id, nouns in per_image_nouns.items()
    }
    return vocabulary, per_image_nouns


def disjoint_by_image_split(
    image_ids: list[int], *, val_fraction: float = 0.02, seed: int = 42,
) -> tuple[list[int], list[int]]:
    """D5: a caption validation split disjoint BY IMAGE from training (no
    image contributes captions to both splits)."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0,1)")
    ids = sorted(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    assert set(val_ids).isdisjoint(train_ids)
    return train_ids, val_ids


def sample_step_vocabulary(
    present_nouns: list[str], vocabulary: list[str], *, n_distractor: int = 32, seed: int | None = None,
) -> tuple[list[str], list[bool]]:
    """D4: per-step C = nouns present in the caption + N_distractor sampled
    ABSENT nouns. Returns (class_list, is_present) with present nouns
    first, distractors after (order matters for L2's ranking loss, which
    needs to know which entries are the "present" positives)."""
    if not present_nouns:
        raise ValueError("sample_step_vocabulary requires at least one present noun")
    present_set = set(present_nouns)
    absent_pool = [w for w in vocabulary if w not in present_set]
    rng = random.Random(seed)
    n_distractor = min(n_distractor, len(absent_pool))
    distractors = rng.sample(absent_pool, n_distractor) if n_distractor > 0 else []
    class_list = list(present_nouns) + distractors
    is_present = [True] * len(present_nouns) + [False] * len(distractors)
    return class_list, is_present
