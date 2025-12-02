"""Helper utilities powering Stage 2 energy sharing simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from django.utils.translation import gettext_lazy as _

from .timeseries import TimeseriesError, parse_member_timeseries

EPSILON = 1e-9


@dataclass
class StageTwoIterationConfig:
    order: int
    key_type: str
    percentages: Dict[int, float]


@dataclass
class MemberAllocationSummary:
    member_id: int
    member_name: str
    total_production_kwh: float
    total_consumption_kwh: float
    community_consumption_kwh: float
    external_consumption_kwh: float
    shared_production_kwh: float
    unused_production_kwh: float


@dataclass
class IterationStatistic:
    order: int
    key_type: str
    allocated_kwh: float


@dataclass
class StageTwoEvaluation:
    member_summaries: List[MemberAllocationSummary]
    iteration_stats: List[IterationStatistic]
    total_community_allocation_kwh: float
    total_remaining_production_kwh: float
    total_unserved_consumption_kwh: float
    timeline: pd.DataFrame
    warnings: List[str]


def load_project_timeseries(members: Sequence) -> Tuple[pd.DataFrame, List[str]]:
    """Return a combined timeseries DataFrame for the provided members."""

    frames: List[pd.DataFrame] = []
    warnings: List[str] = []
    timestamps: set = set()

    for member in members:
        dataset = getattr(member, "dataset", None)
        if not dataset or not getattr(dataset, "normalized_file", None):
            warnings.append(_("%(member)s : aucun dataset normalisé n'est associé.") % {"member": member.name})
            continue

        try:
            parse_result = parse_member_timeseries(dataset.normalized_file.path)
        except (TimeseriesError, FileNotFoundError) as exc:
            warnings.append(f"{member.name} : {exc}")
            continue

        frame = parse_result.data.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
        if frame.empty:
            warnings.append(_("%(member)s : aucune donnée temporelle exploitable.") % {"member": member.name})
            continue

        frame = frame.groupby("timestamp")[["production_kwh", "consumption_kwh"]].sum()
        frame = frame.sort_index()
        timestamps.update(frame.index)
        frame = frame.rename(columns={"production_kwh": "production", "consumption_kwh": "consumption"})
        frames.append((member.id, member.name, frame))
        for note in parse_result.metadata.warnings:
            warnings.append(f"{member.name} : {note}")

    if not frames:
        raise ValueError("Aucune donnée de série temporelle valide n'a été trouvée pour ce projet.")

    full_index = sorted(timestamps)
    aligned: List[pd.DataFrame] = []
    for member_id, member_name, frame in frames:
        reindexed = frame.reindex(full_index, fill_value=0.0)
        reindexed = reindexed.reset_index().rename(columns={"index": "timestamp"})
        reindexed["member_id"] = member_id
        reindexed["member_name"] = member_name
        aligned.append(reindexed)

    combined = pd.concat(aligned, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined = combined.dropna(subset=["timestamp"])
    combined = combined.sort_values(["timestamp", "member_name"])
    combined[["production", "consumption"]] = combined[["production", "consumption"]].astype(float)
    return combined, warnings


def build_iteration_configs(iterations: Iterable[dict], members: Sequence) -> List[StageTwoIterationConfig]:
    member_ids = {member.id for member in members}
    configs: List[StageTwoIterationConfig] = []
    for index, payload in enumerate(iterations, start=1):
        if not isinstance(payload, dict):
            continue
        key_type = payload.get("key_type")
        if key_type not in {"equal", "percentage", "proportional"}:
            continue
        order = payload.get("order") or index
        try:
            order = int(order)
        except (TypeError, ValueError):
            order = index

        raw_percentages = payload.get("percentages") or {}
        percentages: Dict[int, float] = {}
        if isinstance(raw_percentages, dict):
            for member_id, value in raw_percentages.items():
                try:
                    member_key = int(member_id)
                except (TypeError, ValueError):
                    continue
                if member_key not in member_ids:
                    continue
                try:
                    percentages[member_key] = float(value)
                except (TypeError, ValueError):
                    continue

        configs.append(StageTwoIterationConfig(order=order, key_type=key_type, percentages=percentages))

    configs.sort(key=lambda item: item.order)
    return configs


def evaluate_sharing(
    timeseries: pd.DataFrame,
    members: Sequence,
    iterations: Sequence[StageTwoIterationConfig],
) -> StageTwoEvaluation:
    if timeseries.empty:
        raise ValueError("Aucune donnée temporelle disponible pour le partage d'énergie.")
    if not iterations:
        raise ValueError("Le scénario doit contenir au moins une itération active.")

    member_lookup = {member.id: member for member in members}
    unique_member_ids = [member.id for member in members]

    stats = {
        member_id: {
            "total_production": 0.0,
            "total_consumption": 0.0,
            "community_consumption": 0.0,
            "external_consumption": 0.0,
            "shared_production": 0.0,
            "unused_production": 0.0,
        }
        for member_id in unique_member_ids
    }

    iteration_totals: Dict[int, float] = {config.order: 0.0 for config in iterations}
    iteration_types: Dict[int, str] = {config.order: config.key_type for config in iterations}
    warnings: List[str] = []

    timeline_records: List[Dict[str, float]] = []

    timeseries = timeseries.sort_values(["timestamp", "member_id"])  # ensure deterministic ordering

    for timestamp, slice_df in timeseries.groupby("timestamp"):
        production = {int(row.member_id): float(row.production) for row in slice_df.itertuples()}
        consumption = {int(row.member_id): float(row.consumption) for row in slice_df.itertuples()}

        for member_id in unique_member_ids:
            production.setdefault(member_id, 0.0)
            consumption.setdefault(member_id, 0.0)

        original_production = production.copy()
        original_consumption = consumption.copy()
        member_allocations = {member_id: 0.0 for member_id in unique_member_ids}
        member_supply = {member_id: 0.0 for member_id in unique_member_ids}

        for config in iterations:
            pool = sum(production.values())
            if pool <= EPSILON:
                break

            weights = _build_weights(config, consumption)
            if not any(weight > EPSILON for weight in weights.values()):
                warnings.append(
                    _("Itération %(order)s (%(type)s) ignorée faute de demande active au pas %(timestamp)s.")
                    % {"order": config.order, "type": config.key_type, "timestamp": timestamp.isoformat()}
                )
                continue

            allocations, unused_pool = _allocate_to_consumers(pool, consumption, weights)
            allocated_total = sum(allocations.values())
            if allocated_total <= EPSILON:
                continue

            supply = _allocate_from_producers(production, allocated_total)
            for member_id, amount in supply.items():
                member_supply[member_id] += amount

            for member_id, amount in allocations.items():
                consumption[member_id] = max(0.0, consumption[member_id] - amount)
                member_allocations[member_id] += amount

            iteration_totals[config.order] = iteration_totals.get(config.order, 0.0) + allocated_total

        record = {
            "timestamp": timestamp,
            "production_total_kwh": sum(original_production.values()),
            "consumption_total_kwh": sum(original_consumption.values()),
            "community_allocated_kwh": sum(member_allocations.values()),
            "remaining_production_kwh": sum(production.values()),
            "remaining_consumption_kwh": sum(consumption.values()),
        }

        for member_id in unique_member_ids:
            stats_member = stats[member_id]
            stats_member["total_production"] += original_production.get(member_id, 0.0)
            stats_member["total_consumption"] += original_consumption.get(member_id, 0.0)
            stats_member["community_consumption"] += member_allocations.get(member_id, 0.0)
            stats_member["external_consumption"] += consumption.get(member_id, 0.0)
            stats_member["shared_production"] += member_supply.get(member_id, 0.0)
            stats_member["unused_production"] += production.get(member_id, 0.0)

            record[f"member_{member_id}_community_kwh"] = member_allocations.get(member_id, 0.0)
            record[f"member_{member_id}_external_kwh"] = consumption.get(member_id, 0.0)
            record[f"member_{member_id}_production_shared_kwh"] = member_supply.get(member_id, 0.0)
            record[f"member_{member_id}_production_unused_kwh"] = production.get(member_id, 0.0)

        timeline_records.append(record)

    member_summaries = [
        MemberAllocationSummary(
            member_id=member_id,
            member_name=getattr(member_lookup.get(member_id), "name", str(member_id)),
            total_production_kwh=values["total_production"],
            total_consumption_kwh=values["total_consumption"],
            community_consumption_kwh=values["community_consumption"],
            external_consumption_kwh=values["external_consumption"],
            shared_production_kwh=values["shared_production"],
            unused_production_kwh=values["unused_production"],
        )
        for member_id, values in stats.items()
    ]

    iteration_stats = [
        IterationStatistic(order=order, key_type=iteration_types.get(order, ""), allocated_kwh=value)
        for order, value in sorted(iteration_totals.items())
    ]

    timeline = pd.DataFrame(timeline_records)
    if not timeline.empty:
        timeline = timeline.sort_values("timestamp")

    return StageTwoEvaluation(
        member_summaries=member_summaries,
        iteration_stats=iteration_stats,
        total_community_allocation_kwh=float(timeline["community_allocated_kwh"].sum()) if not timeline.empty else 0.0,
        total_remaining_production_kwh=float(timeline["remaining_production_kwh"].sum()) if not timeline.empty else 0.0,
        total_unserved_consumption_kwh=float(timeline["remaining_consumption_kwh"].sum()) if not timeline.empty else 0.0,
        timeline=timeline,
        warnings=warnings,
    )


def _build_weights(config: StageTwoIterationConfig, consumption: Dict[int, float]) -> Dict[int, float]:
    weights: Dict[int, float] = {}
    if config.key_type == "equal":
        for member_id, demand in consumption.items():
            weights[member_id] = 1.0 if demand > EPSILON else 0.0
    elif config.key_type == "percentage":
        weights = {member_id: config.percentages.get(member_id, 0.0) for member_id in consumption.keys()}
    else:  # proportional
        for member_id, demand in consumption.items():
            weights[member_id] = demand if demand > EPSILON else 0.0
    return weights


def _allocate_to_consumers(
    pool: float,
    consumption: Dict[int, float],
    weights: Dict[int, float],
) -> Tuple[Dict[int, float], float]:
    allocations = {member_id: 0.0 for member_id in consumption.keys()}
    remaining_pool = float(pool)
    active = {member_id for member_id, demand in consumption.items() if demand > EPSILON and weights.get(member_id, 0.0) > EPSILON}

    while remaining_pool > EPSILON and active:
        weight_sum = sum(weights.get(member_id, 0.0) for member_id in active)
        if weight_sum <= EPSILON:
            break

        distributed = 0.0
        to_remove = []
        for member_id in list(active):
            share = remaining_pool * (weights.get(member_id, 0.0) / weight_sum)
            demand_remaining = consumption[member_id] - allocations[member_id]
            if demand_remaining <= EPSILON:
                to_remove.append(member_id)
                continue
            allocation = min(share, demand_remaining)
            if allocation <= EPSILON:
                to_remove.append(member_id)
                continue
            allocations[member_id] += allocation
            distributed += allocation
            demand_remaining -= allocation
            if demand_remaining <= EPSILON:
                to_remove.append(member_id)

        for member_id in to_remove:
            active.discard(member_id)

        if distributed <= EPSILON:
            break
        remaining_pool -= distributed

    return allocations, remaining_pool


def _allocate_from_producers(production: Dict[int, float], required: float) -> Dict[int, float]:
    supplies = {member_id: 0.0 for member_id in production.keys()}
    remaining = float(required)
    active = {member_id for member_id, value in production.items() if value > EPSILON}

    while remaining > EPSILON and active:
        total_available = sum(production[member_id] for member_id in active)
        if total_available <= EPSILON:
            break

        distributed = 0.0
        depleted: List[int] = []
        for member_id in list(active):
            available = production[member_id]
            if available <= EPSILON:
                depleted.append(member_id)
                continue
            share = remaining * (available / total_available)
            taken = min(share, available)
            if taken <= EPSILON:
                depleted.append(member_id)
                continue
            supplies[member_id] += taken
            production[member_id] -= taken
            distributed += taken
            if production[member_id] <= EPSILON:
                production[member_id] = 0.0
                depleted.append(member_id)

        for member_id in depleted:
            active.discard(member_id)

        if distributed <= EPSILON:
            break
        remaining -= distributed

    return supplies
