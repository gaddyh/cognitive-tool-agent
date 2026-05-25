from __future__ import annotations

import random
from collections import defaultdict
from typing import Literal

from ..schemas.simulation_profile import SimulationProfile

SplitName = Literal["train", "dev", "test"]

_SPLITS: list[SplitName] = ["train", "dev", "test"]


def stratified_split(
    profiles: list[SimulationProfile],
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> dict[str, SplitName]:
    """Return a mapping of simulation_id → split name.

    Stratification key: (scenario_type, grounding_flag).
    Small buckets (< 3) are handled with global-quota-aware assignment so the
    final totals land at exactly the target counts for the given dataset size.
    """
    n = len(profiles)
    if n == 0:
        return {}

    train_ratio, dev_ratio, _ = ratios
    targets: dict[SplitName, int] = {
        "train": round(n * train_ratio),
        "dev": round(n * dev_ratio),
        "test": n - round(n * train_ratio) - round(n * dev_ratio),
    }

    rng = random.Random(seed)

    buckets: dict[tuple[str, str], list[SimulationProfile]] = defaultdict(list)
    for p in profiles:
        grounding_flag = "grounding" if p.requires_grounding else "no_grounding"
        key = (p.scenario_type, grounding_flag)
        buckets[key].append(p)

    assignments: dict[str, SplitName] = {}
    counts: dict[SplitName, int] = {"train": 0, "dev": 0, "test": 0}

    for key in sorted(buckets.keys()):
        bucket = buckets[key]
        rng.shuffle(bucket)
        bucket_n = len(bucket)

        if bucket_n >= 3:
            n_train = round(bucket_n * train_ratio)
            n_dev = round(bucket_n * dev_ratio)
            n_test = bucket_n - n_train - n_dev
            alloc: list[SplitName] = (
                ["train"] * n_train + ["dev"] * n_dev + ["test"] * n_test
            )
        else:
            alloc = []
            for profile in bucket:
                deficit = {
                    split: targets[split] - counts[split]
                    for split in _SPLITS
                }
                chosen = max(deficit, key=lambda s: deficit[s])
                alloc.append(chosen)

        for profile, split in zip(bucket, alloc):
            assignments[profile.simulation_id] = split
            counts[split] += 1

    return assignments


def assign_splits(
    profiles: list[SimulationProfile],
    assignments: dict[str, SplitName],
) -> list[SimulationProfile]:
    """Return a new list of profiles with split field populated."""
    result = []
    for p in profiles:
        split = assignments.get(p.simulation_id)
        result.append(p.model_copy(update={"split": split}))
    return result
