#!/usr/bin/env python3
"""V1 pilot runner CLI. Trains LearnedMetric for a short schedule,
checkpointing and evaluating (held-out caption losses + full COCO-Stuff val
mIoU via the converged CGLS solve) at fixed intervals, then reports V2's
correlations and V3/V4's checkpoint selection."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/project/6114407/haree/Talk2DINO")

from src.learned_affinity.losses import LossWeights
from src.learned_affinity.pilot import run_pilot
from src.learned_affinity.train import TrainConfig, setup_training_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--total-steps", type=int, default=400)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vocab-image-subset", type=int, default=None)
    parser.add_argument("--n-held-out-samples", type=int, default=50)
    parser.add_argument("--full-val-max-images", type=int, default=None,
                         help="Cap E1-style full-val eval to N images per checkpoint "
                              "(full 5000 costs ~21 min EACH -- use this to fit a budget)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = TrainConfig(
        lr=args.lr, total_steps=args.total_steps, checkpoint_every=args.checkpoint_every,
        batch_size=args.batch_size, seed=args.seed, loss_weights=LossWeights(),
    )
    model, vocab, train_dataset, held_out_dataset = setup_training_data(
        config, output_dir=args.output_dir, device=args.device, vocab_image_subset=args.vocab_image_subset,
    )
    print(f"train_dataset={len(train_dataset)} images, held_out_dataset={len(held_out_dataset)} images")

    run_pilot(
        config, output_dir=args.output_dir, device=args.device,
        held_out_dataset=held_out_dataset, train_dataset=train_dataset, model=model, vocab=vocab,
        n_held_out_samples=args.n_held_out_samples, full_val_max_images=args.full_val_max_images,
    )


if __name__ == "__main__":
    main()
