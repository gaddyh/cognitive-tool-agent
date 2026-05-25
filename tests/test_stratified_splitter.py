"""Unit tests for stratified_splitter — disjoint + quota invariants."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from cognitive_tool_agent.schemas.simulation_profile import SimulationProfile
from cognitive_tool_agent.trace_converter.stratified_splitter import (
    assign_splits,
    stratified_split,
)


def _profile(
    sim_id: str,
    scenario_type: str = "cancel|single_action",
    requires_grounding: bool = True,
    has_item_ids: bool = False,
    has_order_id: bool = True,
    has_product_id: bool = False,
) -> SimulationProfile:
    return SimulationProfile(
        simulation_id=sim_id,
        task_id=sim_id,
        split=None,
        primary_scenario=scenario_type.split("|")[0],
        scenario_type=scenario_type,
        is_multi_action=scenario_type.endswith("multi_action"),
        terminal_tool_fingerprint="cancel_pending_order",
        requires_grounding=requires_grounding,
        requires_tool_chaining=True,
        has_item_ids=has_item_ids,
        has_order_id=has_order_id,
        has_product_id=has_product_id,
        difficulty_bucket="medium",
        num_expected_actions=3,
        num_tool_calls=4,
    )


def _make_profiles(n: int, scenario_type: str = "cancel|single_action", offset: int = 0) -> list[SimulationProfile]:
    return [_profile(f"sim-{i + offset:03d}", scenario_type=scenario_type) for i in range(n)]


class TestDisjoint:
    def test_no_overlap_100_simulations(self):
        profiles = _make_profiles(100)
        assignments = stratified_split(profiles)
        train = {k for k, v in assignments.items() if v == "train"}
        dev = {k for k, v in assignments.items() if v == "dev"}
        test = {k for k, v in assignments.items() if v == "test"}
        assert train.isdisjoint(dev)
        assert train.isdisjoint(test)
        assert dev.isdisjoint(test)

    def test_every_sim_assigned(self):
        profiles = _make_profiles(50)
        assignments = stratified_split(profiles)
        assert len(assignments) == 50
        assert all(v in {"train", "dev", "test"} for v in assignments.values())


class TestExactQuotas:
    def test_exact_60_20_20_for_100(self):
        profiles = _make_profiles(100)
        assignments = stratified_split(profiles)
        counts = {"train": 0, "dev": 0, "test": 0}
        for v in assignments.values():
            counts[v] += 1
        assert abs(counts["train"] - 60) <= 2
        assert abs(counts["dev"] - 20) <= 2
        assert abs(counts["test"] - 20) <= 2

    def test_50_simulations_ratio(self):
        profiles = _make_profiles(50)
        assignments = stratified_split(profiles)
        counts = {"train": 0, "dev": 0, "test": 0}
        for v in assignments.values():
            counts[v] += 1
        assert abs(counts["train"] - 30) <= 2
        assert abs(counts["dev"] - 10) <= 2
        assert abs(counts["test"] - 10) <= 2


class TestStratification:
    def test_multiple_buckets_still_disjoint(self):
        profiles = (
            _make_profiles(40, "cancel|single_action", offset=0)
            + _make_profiles(30, "exchange|single_action", offset=40)
            + _make_profiles(30, "return|multi_action", offset=70)
        )
        assignments = stratified_split(profiles)
        train = {k for k, v in assignments.items() if v == "train"}
        dev = {k for k, v in assignments.items() if v == "dev"}
        test = {k for k, v in assignments.items() if v == "test"}
        assert train.isdisjoint(dev)
        assert train.isdisjoint(test)
        assert dev.isdisjoint(test)
        assert len(assignments) == 100

    def test_single_item_bucket_gets_assigned(self):
        profiles = [
            _profile("rare-sim", scenario_type="transfer|single_action", requires_grounding=False),
        ] + _make_profiles(99, offset=1)
        assignments = stratified_split(profiles)
        assert "rare-sim" in assignments
        assert assignments["rare-sim"] in {"train", "dev", "test"}

    def test_tiny_bucket_does_not_break_global_quota(self):
        profiles = (
            [_profile(f"rare-{i}", scenario_type="transfer|single_action", requires_grounding=False)
             for i in range(2)]
            + _make_profiles(98, offset=2)
        )
        assignments = stratified_split(profiles)
        counts = {"train": 0, "dev": 0, "test": 0}
        for v in assignments.values():
            counts[v] += 1
        assert abs(counts["train"] - 60) <= 2
        assert abs(counts["dev"] - 20) <= 2
        assert abs(counts["test"] - 20) <= 2


class TestDeterminism:
    def test_same_seed_same_result(self):
        profiles = _make_profiles(100)
        a1 = stratified_split(profiles, seed=42)
        a2 = stratified_split(profiles, seed=42)
        assert a1 == a2

    def test_different_seed_different_result(self):
        profiles = _make_profiles(100)
        a1 = stratified_split(profiles, seed=42)
        a2 = stratified_split(profiles, seed=99)
        assert a1 != a2


class TestAssignSplits:
    def test_assign_splits_enriches_profiles(self):
        profiles = _make_profiles(10)
        assignments = stratified_split(profiles)
        enriched = assign_splits(profiles, assignments)
        assert all(p.split is not None for p in enriched)
        assert all(p.split in {"train", "dev", "test"} for p in enriched)

    def test_original_profiles_unchanged(self):
        profiles = _make_profiles(10)
        assignments = stratified_split(profiles)
        assign_splits(profiles, assignments)
        assert all(p.split is None for p in profiles)


class TestEmptyInput:
    def test_empty_returns_empty(self):
        assert stratified_split([]) == {}
