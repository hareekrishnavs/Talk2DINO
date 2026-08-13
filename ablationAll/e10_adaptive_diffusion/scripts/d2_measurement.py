#!/usr/bin/env python3
"""D2 (mandatory): measure DINO forward wall-clock per training sample vs
the CG solve wall-clock, on REAL data. Also exercises the full D1-D4 path
end to end for the first time (real COCO image, real spaCy noun
extraction, real frozen CLIP text encoding, real DINOv2 patch extraction)
as an integration smoke test of everything built so far."""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/project/6114407/haree/Talk2DINO")
sys.path.insert(0, "/project/6114407/haree/Talk2DINO/src/open_vocabulary_segmentation")

import os
os.chdir("/project/6114407/haree/Talk2DINO")
import main  # noqa: F401 -- registers FloatImage pipeline transform

import random
import torch
from models import build_model
import spacy

from src.learned_affinity.coco_captions import (
    build_noun_vocabulary, disjoint_by_image_split, load_captions, sample_step_vocabulary,
)
from src.learned_affinity.coco_captions import COCO_IMAGES_TRAIN
from src.learned_affinity.crop_dataset import CocoCaptionCropDataset
from src.learned_affinity.extract import extract_training_sample
from src.learned_affinity.metric import LearnedMetric, build_differentiable_knn_graph
from src.learned_affinity.implicit_solve import implicit_propagate, LAST_SOLVE_STATS
from src.learned_affinity.text_vocab import (
    VocabularyEmbeddings, build_merged_config, load_or_build_vocabulary_embeddings,
)

print("=== loading captions + building noun vocabulary (real COCO train2017) ===")
t0 = time.perf_counter()
images = load_captions()
print(f"loaded {len(images)} images with captions in {time.perf_counter()-t0:.1f}s")

nlp = spacy.load("en_core_web_sm")
t0 = time.perf_counter()
# small subset for the vocabulary-building timing itself (full corpus done separately, not timed here)
subset_ids = random.Random(0).sample(list(images.keys()), 2000)
subset = {i: images[i] for i in subset_ids}
vocabulary, per_image_nouns = build_noun_vocabulary(subset, nlp, min_count=3)
print(f"built vocabulary of {len(vocabulary)} nouns from {len(subset)} images in {time.perf_counter()-t0:.1f}s")

train_ids, val_ids = disjoint_by_image_split(list(subset.keys()), val_fraction=0.05, seed=0)
print(f"train/val split: {len(train_ids)} / {len(val_ids)} images (disjoint by image)")

print("\n=== loading frozen model (same pattern as capture_dino_features.py) ===")
cfg = build_merged_config()
model = build_model(cfg.model)
model.cuda()
model.eval()
print("template:", cfg.evaluate.template)

print("\n=== encoding noun vocabulary once (frozen CLIP text encoder) ===")
t0 = time.perf_counter()
vocab_cache = Path("/scratch/haree/talk2dino_e3_affinity_oracle/verify_e10/noun_vocab_smoke.pt")
payload = load_or_build_vocabulary_embeddings(vocab_cache, model, vocabulary, cfg.evaluate.template, overwrite=True)
print(f"encoded {len(vocabulary)} nouns in {time.perf_counter()-t0:.1f}s -> {payload['embeddings'].shape}")
vocab = VocabularyEmbeddings(payload["vocabulary"], payload["embeddings"])

file_names = {i: images[i]["file_name"] for i in subset_ids}
dataset = CocoCaptionCropDataset(COCO_IMAGES_TRAIN, train_ids, file_names, per_image_nouns, seed=0)
print(f"train dataset (after noun-filtering): {len(dataset)} images")

print("\n=== D2: measuring DINO forward vs CG solve wall-clock, 20 real samples ===")
metric = LearnedMetric().cuda()
device = torch.device("cuda")

dino_times = []
solve_fwd_times = []
solve_bwd_times = []
n_samples = 20
for i in range(n_samples):
    item = dataset[i % len(dataset)]
    class_list, is_present = sample_step_vocabulary(item["present_nouns"], vocabulary, n_distractor=32, seed=i)
    text_embedding = vocab.gather(class_list, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    features, raw_scores = extract_training_sample(model, item["crop_bgr"], text_embedding)
    torch.cuda.synchronize()
    dino_times.append(time.perf_counter() - t0)

    g = metric(features)
    indices, weights = build_differentiable_knn_graph(g, k=metric.k, kappa=metric.kappa)
    s0 = raw_scores.T.contiguous()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    s_star = implicit_propagate(s0, indices, weights, 0.98)
    torch.cuda.synchronize()
    solve_fwd_times.append(time.perf_counter() - t0)

    loss = s_star.sum()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    solve_bwd_times.append(time.perf_counter() - t0)
    metric.zero_grad()

def stats(name, values):
    mean_ms = sum(values) / len(values) * 1000
    print(f"{name}: mean={mean_ms:.2f}ms  min={min(values)*1000:.2f}ms  max={max(values)*1000:.2f}ms")

print(f"\n(first sample excluded from stats below -- CUDA/cuDNN warm-up)")
stats("DINO forward (extract_training_sample: crop -> features+S0)", dino_times[1:])
stats("CG forward solve", solve_fwd_times[1:])
stats("CG backward (adjoint) solve", solve_bwd_times[1:])

dino_mean = sum(dino_times[1:]) / len(dino_times[1:])
solve_mean = sum(solve_fwd_times[1:]) / len(solve_fwd_times[1:]) + sum(solve_bwd_times[1:]) / len(solve_bwd_times[1:])
ratio = solve_mean / dino_mean
print(f"\nCG solve (fwd+bwd) / DINO forward ratio: {ratio:.2f}x")
if dino_mean > 2 * solve_mean:
    print("DINO forward DOMINATES by >2x -- caching a fixed-crop subset would help; see D2 cost report below.")
elif solve_mean > 2 * dino_mean:
    print("CG SOLVE dominates by >2x -- on-the-fly DINO extraction is NOT the bottleneck; caching would not help training throughput.")
else:
    print("Neither dominates by >2x -- roughly comparable cost.")
