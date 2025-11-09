"""Stage 3 computation helpers.

This module gathers pure-Python data structures and utilities to keep
business logic separate from Django views/forms.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Iterable, Any

OBJECTIVE_NONE = "none"
OBJECTIVE_MINIMISE_MEMBER = "minimize_member_cost"
OBJECTIVE_MINIMISE_TOTAL = "minimize_total_cost"
OBJECTIVE_MINIMISE_AVG = "minimize_average_cost_per_kwh"

OBJECTIVE_CHOICES = [
    (OBJECTIVE_NONE, "No optimization (use provided inputs)"),
    (OBJECTIVE_MINIMISE_MEMBER, "Minimize the cost for a specific member"),
    (OBJECTIVE_MINIMISE_TOTAL, "Minimize the total annual community cost"),
    (OBJECTIVE_MINIMISE_AVG, "Minimize the average cost per kWh"),
]


@dataclass(frozen=True)
class TariffComponents:
    distribution_eur_per_kwh: float
    transport_eur_per_kwh: float
    green_surcharge_eur_per_kwh: float
    connection_eur_per_kwh: float
    excise_tax_eur_per_kwh: float
    federal_tax_eur_per_kwh: float
    community_management_eur_per_kwh: float

    @classmethod
    def from_per_mwh(
        cls,
        *,
        distribution_eur_per_mwh: float,
        transport_eur_per_mwh: float,
        green_surcharge_eur_per_mwh: float,
        connection_eur_per_mwh: float,
        excise_tax_eur_per_mwh: float,
        federal_tax_eur_per_mwh: float,
        community_management_eur_per_mwh: float,
    ) -> "TariffComponents":
        def _to_per_kwh(value: float) -> float:
            return float(value or 0.0) / 1000.0

        return cls(
            distribution_eur_per_kwh=_to_per_kwh(distribution_eur_per_mwh),
            transport_eur_per_kwh=_to_per_kwh(transport_eur_per_mwh),
            green_surcharge_eur_per_kwh=_to_per_kwh(green_surcharge_eur_per_mwh),
            connection_eur_per_kwh=_to_per_kwh(connection_eur_per_mwh),
            excise_tax_eur_per_kwh=_to_per_kwh(excise_tax_eur_per_mwh),
            federal_tax_eur_per_kwh=_to_per_kwh(federal_tax_eur_per_mwh),
            community_management_eur_per_kwh=_to_per_kwh(community_management_eur_per_mwh),
        )

    def traditional_unit_cost(self, energy_price_eur_per_kwh: float) -> float:
        return (
            max(energy_price_eur_per_kwh, 0.0)
            + self.distribution_eur_per_kwh
            + self.transport_eur_per_kwh
            + self.green_surcharge_eur_per_kwh
            + self.connection_eur_per_kwh
            + self.excise_tax_eur_per_kwh
            + self.federal_tax_eur_per_kwh
        )

    def community_unit_cost(self, community_energy_price_eur_per_kwh: float) -> float:
        return self.traditional_unit_cost(community_energy_price_eur_per_kwh) + self.community_management_eur_per_kwh


DEFAULT_TARIFFS = TariffComponents.from_per_mwh(
    distribution_eur_per_mwh=101.8,
    transport_eur_per_mwh=28.1,
    green_surcharge_eur_per_mwh=28.38,
    connection_eur_per_mwh=0.75,
    excise_tax_eur_per_mwh=14.21,
    federal_tax_eur_per_mwh=1.92,
    community_management_eur_per_mwh=9.0,
)


@dataclass
class MemberFinancials:
    member_id: int
    name: str
    annual_consumption_kwh: float
    current_unit_price_eur_per_kwh: float
    current_fixed_annual_fee_eur: float
    injected_energy_kwh: float
    injection_price_eur_per_kwh: float
    tariffs: TariffComponents = DEFAULT_TARIFFS

    @property
    def injection_revenue(self) -> float:
        return max(self.injected_energy_kwh, 0.0) * max(self.injection_price_eur_per_kwh, 0.0)

    @property
    def traditional_unit_cost(self) -> float:
        return self.tariffs.traditional_unit_cost(self.current_unit_price_eur_per_kwh)

    @property
    def current_cost(self) -> float:
        energy_cost = max(self.annual_consumption_kwh, 0.0) * self.traditional_unit_cost
        return energy_cost + max(self.current_fixed_annual_fee_eur, 0.0) - self.injection_revenue

    @property
    def current_cost_per_kwh(self) -> Optional[float]:
        if self.annual_consumption_kwh > 0:
            return self.current_cost / self.annual_consumption_kwh
        return None

    def with_tariffs(self, tariffs: TariffComponents) -> "MemberFinancials":
        return replace(self, tariffs=tariffs)

    def build_overview(self) -> "MemberFinancialOverview":
        return MemberFinancialOverview(
            member_id=self.member_id,
            name=self.name,
            annual_consumption_kwh=self.annual_consumption_kwh,
            energy_price_component_eur_per_kwh=self.current_unit_price_eur_per_kwh,
            traditional_unit_cost_eur_per_kwh=self.traditional_unit_cost,
            current_fixed_annual_fee_eur=self.current_fixed_annual_fee_eur,
            injected_energy_kwh=self.injected_energy_kwh,
            injection_price_eur_per_kwh=self.injection_price_eur_per_kwh,
            current_cost=self.current_cost,
            current_cost_per_kwh=self.current_cost_per_kwh,
        )

    @classmethod
    def from_member(cls, member: Any) -> "MemberFinancials":
        consumption = float(member.annual_consumption_kwh or 0.0)
        return cls(
            member_id=member.id,
            name=member.name,
            annual_consumption_kwh=consumption,
            current_unit_price_eur_per_kwh=float(member.current_unit_price_eur_per_kwh or 0.0),
            current_fixed_annual_fee_eur=float(member.current_fixed_annual_fee_eur or 0.0),
            injected_energy_kwh=float(member.injected_energy_kwh or 0.0),
            injection_price_eur_per_kwh=float(member.injection_price_eur_per_kwh or 0.0),
            tariffs=DEFAULT_TARIFFS,
        )


@dataclass
class CommunityScenarioParameters:
    name: str
    community_price: float
    allow_price_range: bool
    price_min: float
    price_max: float
    price_step: Optional[float]
    total_community_fee: float
    per_member_fee: float
    coverage_targets: Dict[int, float]  # ratio between 0 and 1
    objective: str = OBJECTIVE_NONE
    target_member_id: Optional[int] = None
    tariffs: TariffComponents = DEFAULT_TARIFFS


@dataclass
class MemberScenarioResult:
    member_id: int
    member_name: str
    coverage_ratio: float
    coverage_percentage: float
    community_energy_kwh: float
    supplier_energy_kwh: float
    current_cost: float
    community_cost: float
    savings: float
    current_cost_per_kwh: Optional[float]
    community_cost_per_kwh: Optional[float]


@dataclass
class MemberFinancialOverview:
    member_id: int
    name: str
    annual_consumption_kwh: float
    energy_price_component_eur_per_kwh: float
    traditional_unit_cost_eur_per_kwh: float
    current_fixed_annual_fee_eur: float
    injected_energy_kwh: float
    injection_price_eur_per_kwh: float
    current_cost: float
    current_cost_per_kwh: Optional[float]


@dataclass
class CommunityScenarioSummary:
    total_current_cost: float
    total_community_cost: float
    total_annual_savings: float
    total_consumption_kwh: float
    average_savings_per_member: float
    average_cost_per_kwh_current: Optional[float]
    average_cost_per_kwh_community: Optional[float]


@dataclass
class StageThreeComputation:
    scenario: CommunityScenarioParameters
    applied_price: float
    coverage_map: Dict[int, float]
    member_results: List[MemberScenarioResult]
    summary: CommunityScenarioSummary
    optimization_details: Dict[str, Any] = field(default_factory=dict)


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, value))


def _allocate_fee(total_fee: float, members: Iterable[MemberFinancials]) -> Dict[int, float]:
    member_list = list(members)
    if not member_list:
        return {}
    share = float(total_fee or 0.0) / len(member_list)
    return {m.member_id: share for m in member_list}


def _compute_member_cost(
    member: MemberFinancials,
    coverage_ratio: float,
    community_price: float,
    community_fee_share: float,
    per_member_fee: float,
) -> MemberScenarioResult:
    coverage_ratio = _clamp_ratio(coverage_ratio)
    consumption = max(member.annual_consumption_kwh, 0.0)
    community_energy_kwh = consumption * coverage_ratio
    supplier_energy_kwh = consumption - community_energy_kwh

    community_unit_cost = member.tariffs.community_unit_cost(community_price)
    supplier_unit_cost = member.traditional_unit_cost

    community_energy_cost = community_energy_kwh * community_unit_cost
    supplier_energy_cost = supplier_energy_kwh * supplier_unit_cost

    community_cost = (
        community_energy_cost
        + supplier_energy_cost
        + max(member.current_fixed_annual_fee_eur, 0.0)
        + max(community_fee_share, 0.0)
        + max(per_member_fee, 0.0)
        - member.injection_revenue
    )

    current_cost = member.current_cost
    savings = current_cost - community_cost

    community_cost_per_kwh = None
    if consumption > 0:
        community_cost_per_kwh = community_cost / consumption

    return MemberScenarioResult(
        member_id=member.member_id,
        member_name=member.name,
        coverage_ratio=coverage_ratio,
        coverage_percentage=coverage_ratio * 100.0,
        community_energy_kwh=community_energy_kwh,
        supplier_energy_kwh=supplier_energy_kwh,
        current_cost=current_cost,
        community_cost=community_cost,
        savings=savings,
        current_cost_per_kwh=member.current_cost_per_kwh,
        community_cost_per_kwh=community_cost_per_kwh,
    )


def _evaluate_members(
    members: List[MemberFinancials],
    scenario: CommunityScenarioParameters,
    price: float,
    coverage_map: Dict[int, float],
) -> Tuple[List[MemberScenarioResult], CommunityScenarioSummary]:
    fee_map = _allocate_fee(scenario.total_community_fee, members)
    member_results: List[MemberScenarioResult] = []
    total_current_cost = 0.0
    total_community_cost = 0.0
    total_consumption = 0.0

    for member in members:
        coverage_ratio = coverage_map.get(member.member_id, scenario.coverage_targets.get(member.member_id, 0.0))
        result = _compute_member_cost(
            member,
            coverage_ratio,
            price,
            fee_map.get(member.member_id, 0.0),
            scenario.per_member_fee,
        )
        member_results.append(result)
        total_current_cost += result.current_cost
        total_community_cost += result.community_cost
        total_consumption += max(member.annual_consumption_kwh, 0.0)

    total_savings = total_current_cost - total_community_cost
    average_savings = total_savings / len(members) if members else 0.0
    average_cost_current = None
    average_cost_community = None
    if total_consumption > 0:
        average_cost_current = total_current_cost / total_consumption
        average_cost_community = total_community_cost / total_consumption

    summary = CommunityScenarioSummary(
        total_current_cost=total_current_cost,
        total_community_cost=total_community_cost,
        total_annual_savings=total_savings,
        total_consumption_kwh=total_consumption,
        average_savings_per_member=average_savings,
        average_cost_per_kwh_current=average_cost_current,
        average_cost_per_kwh_community=average_cost_community,
    )

    return member_results, summary


def _generate_price_candidates(scenario: CommunityScenarioParameters) -> List[float]:
    if not scenario.allow_price_range:
        return [scenario.community_price]

    start = min(scenario.price_min, scenario.price_max)
    end = max(scenario.price_min, scenario.price_max)
    if start == end:
        return [start]

    step = scenario.price_step or (end - start) / 10.0 or 0.01
    if step <= 0:
        step = (end - start) / 10.0 or 0.01

    prices: List[float] = []
    current = start
    max_iterations = 1000
    iterations = 0
    while current <= end + 1e-9 and iterations < max_iterations:
        prices.append(round(current, 6))
        current += step
        iterations += 1
    if prices[-1] != round(end, 6):
        prices.append(round(end, 6))
    return prices


def run_stage_three(members: List[MemberFinancials], scenario: CommunityScenarioParameters) -> Optional[StageThreeComputation]:
    if not members:
        return None

    members_with_tariffs = [member.with_tariffs(scenario.tariffs) for member in members]

    if scenario.objective == OBJECTIVE_NONE:
        price = scenario.community_price
        coverage_map = {mid: _clamp_ratio(r) for mid, r in scenario.coverage_targets.items()}
        member_results, summary = _evaluate_members(members_with_tariffs, scenario, price, coverage_map)
        return StageThreeComputation(
            scenario=scenario,
            applied_price=price,
            coverage_map=coverage_map,
            member_results=member_results,
            summary=summary,
            optimization_details={"objective": OBJECTIVE_NONE},
        )

    return _run_optimization(members_with_tariffs, scenario)


def _run_optimization(
    members: List[MemberFinancials],
    scenario: CommunityScenarioParameters,
) -> Optional[StageThreeComputation]:
    price_candidates = _generate_price_candidates(scenario)
    best_result: Optional[StageThreeComputation] = None
    best_metric: Optional[float] = None
    fee_map = _allocate_fee(scenario.total_community_fee, members)

    if scenario.objective == OBJECTIVE_MINIMISE_MEMBER:
        target_id = scenario.target_member_id or (members[0].member_id if members else None)
        if target_id is None:
            return None

        for price in price_candidates:
            max_ratio = scenario.coverage_targets.get(target_id, 1.0)
            for ratio in (0.0, max_ratio):
                coverage_map = {mid: _clamp_ratio(r) for mid, r in scenario.coverage_targets.items()}
                coverage_map[target_id] = _clamp_ratio(ratio)
                member_results, summary = _evaluate_members(members, scenario, price, coverage_map)
                target_result = next((r for r in member_results if r.member_id == target_id), None)
                if target_result is None:
                    continue
                metric = target_result.community_cost
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    best_result = StageThreeComputation(
                        scenario=scenario,
                        applied_price=price,
                        coverage_map=coverage_map,
                        member_results=member_results,
                        summary=summary,
                        optimization_details={
                            "objective": OBJECTIVE_MINIMISE_MEMBER,
                            "target_member_id": target_id,
                            "evaluated_prices": price_candidates,
                        },
                    )
                    best_result.optimization_details["metric_value"] = metric
                    best_result.optimization_details["metric_label"] = "Annual community bill for focused member (€)"
        return best_result

    # For total/average objectives evaluate extreme coverage for each member (0 or provided target ratio)
    for price in price_candidates:
        coverage_map = {}
        for member in members:
            max_ratio = scenario.coverage_targets.get(member.member_id, 1.0)
            candidate_ratios = (0.0, _clamp_ratio(max_ratio))
            best_member_ratio = max_ratio
            best_member_cost = None
            for ratio in candidate_ratios:
                result = _compute_member_cost(
                    member,
                    ratio,
                    price,
                    fee_map.get(member.member_id, 0.0),
                    scenario.per_member_fee,
                )
                if best_member_cost is None or result.community_cost < best_member_cost:
                    best_member_cost = result.community_cost
                    best_member_ratio = ratio
            coverage_map[member.member_id] = _clamp_ratio(best_member_ratio)

        member_results, summary = _evaluate_members(members, scenario, price, coverage_map)
        if scenario.objective == OBJECTIVE_MINIMISE_TOTAL:
            metric = summary.total_community_cost
        else:
            metric = summary.average_cost_per_kwh_community or float("inf")

        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_result = StageThreeComputation(
                scenario=scenario,
                applied_price=price,
                coverage_map=coverage_map,
                member_results=member_results,
                summary=summary,
                optimization_details={
                    "objective": scenario.objective,
                    "evaluated_prices": price_candidates,
                },
            )
            if scenario.objective == OBJECTIVE_MINIMISE_TOTAL:
                best_result.optimization_details["metric_label"] = "Total annual community cost (€)"
            else:
                best_result.optimization_details["metric_label"] = "Average community cost per kWh (€/kWh)"
            best_result.optimization_details["metric_value"] = metric

    return best_result
