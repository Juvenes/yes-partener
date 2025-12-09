"""Helper objects and algorithms for Stage 3 cost optimisation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass
class MemberTariff:
    member_id: int
    name: str
    utility: str
    supplier_energy_price_eur_per_kwh: float
    distribution_tariff_eur_per_kwh: float
    transport_tariff_eur_per_kwh: float
    green_support_eur_per_kwh: float
    access_fee_eur_per_kwh: float
    special_excise_eur_per_kwh: float
    energy_contribution_eur_per_kwh: float
    injection_price_eur_per_kwh: float

    @property
    def total_unit_price_eur_per_kwh(self) -> float:
        return round(
            self.supplier_energy_price_eur_per_kwh
            + self.distribution_tariff_eur_per_kwh
            + self.transport_tariff_eur_per_kwh
            + self.green_support_eur_per_kwh
            + self.access_fee_eur_per_kwh
            + self.special_excise_eur_per_kwh
            + self.energy_contribution_eur_per_kwh,
            6,
        )

    @property
    def total_unit_price_eur_per_mwh(self) -> float:
        return round(self.total_unit_price_eur_per_kwh * 1000.0, 3)


@dataclass
class BaselineResult:
    member_id: int
    name: str
    utility: str
    consumption_kwh: float
    production_kwh: float
    grid_import_kwh: float
    grid_export_kwh: float
    unit_price_eur_per_kwh: float
    cost_eur: float


@dataclass
class FlowBreakdown:
    member_id: int
    community_import_kwh: float = 0.0
    community_export_kwh: float = 0.0
    grid_import_kwh: float = 0.0
    grid_export_kwh: float = 0.0


@dataclass
class CommunityOutcome:
    member_id: int
    name: str
    utility: str
    baseline_cost_eur: float
    community_cost_eur: float
    delta_eur: float
    savings_pct: float
    baseline_unit_price_eur_per_kwh: float
    community_unit_price_eur_per_kwh: float
    consumption_kwh: float
    community_import_kwh: float
    grid_import_kwh: float
    community_export_kwh: float
    grid_export_kwh: float


@dataclass
class OptimizationSummary:
    optimal_price_eur_per_kwh: Optional[float]
    candidates_tested: int
    feasible_candidates: int
    min_injection: Optional[float]
    max_supplier: Optional[float]
    member_outcomes: List[CommunityOutcome]
    total_baseline_cost: float
    total_community_cost: float
    total_delta: float


def build_member_tariffs(members: Sequence) -> List[MemberTariff]:
    tariffs: List[MemberTariff] = []
    for member in members:
        tariffs.append(
            MemberTariff(
                member_id=member.id,
                name=member.name,
                utility=getattr(member, "utility", ""),
                supplier_energy_price_eur_per_kwh=float(
                    getattr(member, "supplier_energy_price_eur_per_kwh", 0.0) or 0.0
                ),
                distribution_tariff_eur_per_kwh=float(
                    getattr(member, "distribution_tariff_eur_per_kwh", 0.0) or 0.0
                ),
                transport_tariff_eur_per_kwh=float(
                    getattr(member, "transport_tariff_eur_per_kwh", 0.0) or 0.0
                ),
                green_support_eur_per_kwh=float(
                    getattr(member, "green_support_eur_per_kwh", 0.0) or 0.0
                ),
                access_fee_eur_per_kwh=float(
                    getattr(member, "access_fee_eur_per_kwh", 0.0) or 0.0
                ),
                special_excise_eur_per_kwh=float(
                    getattr(member, "special_excise_eur_per_kwh", 0.0) or 0.0
                ),
                energy_contribution_eur_per_kwh=float(
                    getattr(member, "energy_contribution_eur_per_kwh", 0.0) or 0.0
                ),
                injection_price_eur_per_kwh=float(
                    getattr(member, "injection_price_eur_per_kwh", 0.0) or 0.0
                ),
            )
        )
    return tariffs


def compute_baselines(timeseries: pd.DataFrame, tariffs: Sequence[MemberTariff]) -> Dict[int, BaselineResult]:
    results: Dict[int, BaselineResult] = {}
    if timeseries is None or timeseries.empty:
        for tariff in tariffs:
            results[tariff.member_id] = BaselineResult(
                member_id=tariff.member_id,
                name=tariff.name,
                utility=tariff.utility,
                consumption_kwh=0.0,
                production_kwh=0.0,
                grid_import_kwh=0.0,
                grid_export_kwh=0.0,
                unit_price_eur_per_kwh=tariff.total_unit_price_eur_per_kwh,
                cost_eur=0.0,
            )
        return results

    for tariff in tariffs:
        member_df = timeseries[timeseries["member_id"] == tariff.member_id]
        if member_df.empty:
            results[tariff.member_id] = BaselineResult(
                member_id=tariff.member_id,
                name=tariff.name,
                utility=tariff.utility,
                consumption_kwh=0.0,
                production_kwh=0.0,
                grid_import_kwh=0.0,
                grid_export_kwh=0.0,
                unit_price_eur_per_kwh=tariff.total_unit_price_eur_per_kwh,
                cost_eur=0.0,
            )
            continue

        consumption = float(member_df["consumption"].sum())
        production = float(member_df["production"].sum())
        grid_import = float((member_df["consumption"] - member_df["production"]).clip(lower=0).sum())
        grid_export = float((member_df["production"] - member_df["consumption"]).clip(lower=0).sum())
        cost = grid_import * tariff.total_unit_price_eur_per_kwh - grid_export * tariff.injection_price_eur_per_kwh
        results[tariff.member_id] = BaselineResult(
            member_id=tariff.member_id,
            name=tariff.name,
            utility=tariff.utility,
            consumption_kwh=consumption,
            production_kwh=production,
            grid_import_kwh=grid_import,
            grid_export_kwh=grid_export,
            unit_price_eur_per_kwh=tariff.total_unit_price_eur_per_kwh,
            cost_eur=cost,
        )
    return results


def compute_flows(timeseries: pd.DataFrame, member_ids: Sequence[int]) -> Dict[int, FlowBreakdown]:
    flows: Dict[int, FlowBreakdown] = {member_id: FlowBreakdown(member_id=member_id) for member_id in member_ids}
    if timeseries is None or timeseries.empty:
        return flows

    consumption = timeseries.pivot_table(index="timestamp", columns="member_id", values="consumption", fill_value=0.0)
    production = timeseries.pivot_table(index="timestamp", columns="member_id", values="production", fill_value=0.0)

    for ts in consumption.index:
        cons_row = consumption.loc[ts]
        prod_row = production.loc[ts]
        net = prod_row - cons_row
        surplus = net[net > 0]
        deficit = -net[net < 0]

        total_surplus = float(surplus.sum())
        total_deficit = float(deficit.sum())
        shared = min(total_surplus, total_deficit)

        for member_id in member_ids:
            d = float(deficit.get(member_id, 0.0)) if total_deficit > 0 else 0.0
            s = float(surplus.get(member_id, 0.0)) if total_surplus > 0 else 0.0

            if d > 0 and total_deficit > 0 and shared > 0:
                community_import = shared * (d / total_deficit)
            else:
                community_import = 0.0

            if s > 0 and total_surplus > 0 and shared > 0:
                community_export = shared * (s / total_surplus)
            else:
                community_export = 0.0

            flows[member_id].community_import_kwh += community_import
            flows[member_id].grid_import_kwh += max(d - community_import, 0.0)
            flows[member_id].community_export_kwh += community_export
            flows[member_id].grid_export_kwh += max(s - community_export, 0.0)

    return flows


def compute_flows_from_stage2(
    evaluation, member_ids: Sequence[int]
) -> Dict[int, FlowBreakdown]:
    """Translate a Stage 2 evaluation timeline into Stage 3 flow breakdowns.

    This keeps Phase 3 strictly aligned with the sharing that was computed in
    Phase 2: the community import/export and residual grid usage are directly
    derived from the timeline columns produced by ``evaluate_sharing``.
    """

    flows: Dict[int, FlowBreakdown] = {member_id: FlowBreakdown(member_id=member_id) for member_id in member_ids}
    if evaluation is None or getattr(evaluation, "timeline", None) is None:
        return flows

    timeline = evaluation.timeline
    if timeline is None or timeline.empty:
        return flows

    for member_id in member_ids:
        community_col = f"member_{member_id}_community_kwh"
        external_col = f"member_{member_id}_external_kwh"
        shared_col = f"member_{member_id}_production_shared_kwh"
        unused_col = f"member_{member_id}_production_unused_kwh"

        flows[member_id].community_import_kwh = float(timeline.get(community_col, pd.Series()).sum())
        flows[member_id].grid_import_kwh = float(timeline.get(external_col, pd.Series()).sum())
        flows[member_id].community_export_kwh = float(timeline.get(shared_col, pd.Series()).sum())
        flows[member_id].grid_export_kwh = float(timeline.get(unused_col, pd.Series()).sum())

    return flows


def _community_unit_price(
    tariff: MemberTariff,
    *,
    community_price_eur_per_kwh: float,
    community_fee_eur_per_kwh: float,
    community_type: str,
    reduced_distribution: Optional[float] = None,
    reduced_transport: Optional[float] = None,
) -> float:
    distribution = tariff.distribution_tariff_eur_per_kwh
    transport = tariff.transport_tariff_eur_per_kwh
    if community_type == "single_building":
        if reduced_distribution is not None:
            distribution = reduced_distribution
        if reduced_transport is not None:
            transport = reduced_transport

    return round(
        community_price_eur_per_kwh
        + community_fee_eur_per_kwh
        + distribution
        + transport
        + tariff.green_support_eur_per_kwh
        + tariff.access_fee_eur_per_kwh
        + tariff.special_excise_eur_per_kwh
        + tariff.energy_contribution_eur_per_kwh,
        6,
    )


def evaluate_candidate(
    *,
    price: float,
    tariffs: Sequence[MemberTariff],
    baselines: Dict[int, BaselineResult],
    flows: Dict[int, FlowBreakdown],
    community_fee_eur_per_kwh: float,
    community_type: str,
    reduced_distribution: Optional[float],
    reduced_transport: Optional[float],
) -> Tuple[List[CommunityOutcome], float, float]:
    outcomes: List[CommunityOutcome] = []
    total_baseline_cost = 0.0
    total_community_cost = 0.0

    for tariff in tariffs:
        baseline = baselines.get(tariff.member_id)
        flow = flows.get(tariff.member_id, FlowBreakdown(member_id=tariff.member_id))
        community_unit_price = _community_unit_price(
            tariff,
            community_price_eur_per_kwh=price,
            community_fee_eur_per_kwh=community_fee_eur_per_kwh,
            community_type=community_type,
            reduced_distribution=reduced_distribution,
            reduced_transport=reduced_transport,
        )

        baseline_cost = baseline.cost_eur if baseline else 0.0
        community_cost = (
            flow.community_import_kwh * community_unit_price
            + flow.grid_import_kwh * tariff.total_unit_price_eur_per_kwh
            - flow.community_export_kwh * price
            - flow.grid_export_kwh * tariff.injection_price_eur_per_kwh
        )
        delta = baseline_cost - community_cost
        savings_pct = 0.0 if baseline_cost == 0 else (delta / baseline_cost) * 100.0

        outcomes.append(
            CommunityOutcome(
                member_id=tariff.member_id,
                name=tariff.name,
                utility=tariff.utility,
                baseline_cost_eur=baseline_cost,
                community_cost_eur=community_cost,
                delta_eur=delta,
                savings_pct=savings_pct,
                baseline_unit_price_eur_per_kwh=tariff.total_unit_price_eur_per_kwh,
                community_unit_price_eur_per_kwh=community_unit_price,
                consumption_kwh=baseline.consumption_kwh if baseline else 0.0,
                community_import_kwh=flow.community_import_kwh,
                grid_import_kwh=flow.grid_import_kwh,
                community_export_kwh=flow.community_export_kwh,
                grid_export_kwh=flow.grid_export_kwh,
            )
        )

        total_baseline_cost += baseline_cost
        total_community_cost += community_cost

    return outcomes, total_baseline_cost, total_community_cost


def optimise_internal_price(
    *,
    tariffs: Sequence[MemberTariff],
    timeseries: Optional[pd.DataFrame],
    community_fee_eur_per_kwh: float,
    community_type: str,
    reduced_distribution: Optional[float] = None,
    reduced_transport: Optional[float] = None,
    baselines: Optional[Dict[int, BaselineResult]] = None,
    flows: Optional[Dict[int, FlowBreakdown]] = None,
) -> Optional[OptimizationSummary]:
    if not tariffs:
        return None

    min_injection = max(t.injection_price_eur_per_kwh for t in tariffs)
    max_supplier = max(t.supplier_energy_price_eur_per_kwh for t in tariffs)

    if min_injection > max_supplier:
        return OptimizationSummary(
            optimal_price_eur_per_kwh=None,
            candidates_tested=0,
            feasible_candidates=0,
            min_injection=min_injection,
            max_supplier=max_supplier,
            member_outcomes=[],
            total_baseline_cost=0.0,
            total_community_cost=0.0,
            total_delta=0.0,
        )

    span = max_supplier - min_injection
    if span <= 0:
        candidates = [round(min_injection, 6)]
    else:
        step = max(span * 0.01, 0.0001)
        current = min_injection
        candidates = []
        while current <= max_supplier + 1e-9:
            candidates.append(round(current, 6))
            current += step
        if candidates[-1] != round(max_supplier, 6):
            candidates.append(round(max_supplier, 6))

    if baselines is None:
        baselines = compute_baselines(timeseries, tariffs)
    if flows is None:
        flows = compute_flows(timeseries, [t.member_id for t in tariffs])

    best_price = None
    best_outcomes: List[CommunityOutcome] = []
    best_totals: Tuple[float, float] = (0.0, 0.0)
    feasible = 0

    for candidate in candidates:
        outcomes, baseline_cost, community_cost = evaluate_candidate(
            price=candidate,
            tariffs=tariffs,
            baselines=baselines,
            flows=flows,
            community_fee_eur_per_kwh=community_fee_eur_per_kwh,
            community_type=community_type,
            reduced_distribution=reduced_distribution,
            reduced_transport=reduced_transport,
        )
        if any(out.delta_eur < -1e-6 for out in outcomes):
            continue
        feasible += 1
        total_delta = baseline_cost - community_cost
        if best_price is None or total_delta > (best_totals[0] - best_totals[1]) + 1e-6:
            best_price = candidate
            best_outcomes = outcomes
            best_totals = (baseline_cost, community_cost)

    if best_price is None:
        return OptimizationSummary(
            optimal_price_eur_per_kwh=None,
            candidates_tested=len(candidates),
            feasible_candidates=feasible,
            min_injection=min_injection,
            max_supplier=max_supplier,
            member_outcomes=[],
            total_baseline_cost=sum(b.cost_eur for b in baselines.values()),
            total_community_cost=0.0,
            total_delta=0.0,
        )

    total_baseline_cost, total_community_cost = best_totals
    return OptimizationSummary(
        optimal_price_eur_per_kwh=best_price,
        candidates_tested=len(candidates),
        feasible_candidates=feasible,
        min_injection=min_injection,
        max_supplier=max_supplier,
        member_outcomes=best_outcomes,
        total_baseline_cost=total_baseline_cost,
        total_community_cost=total_community_cost,
        total_delta=total_baseline_cost - total_community_cost,
    )
