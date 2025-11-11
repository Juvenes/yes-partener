"""Helper objects and algorithms powering Stage 3 cost simulations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from django.utils.functional import cached_property


@dataclass
class MemberCostInput:
    member_id: int
    name: str
    consumption_kwh: float
    current_price_eur_per_kwh: float
    current_fixed_fee_eur: float
    injection_kwh: float
    injection_price_eur_per_kwh: float
    utility: str = ""

    def current_cost(self) -> float:
        energy_cost = self.consumption_kwh * self.current_price_eur_per_kwh
        injection_revenue = self.injection_kwh * self.injection_price_eur_per_kwh
        return energy_cost + self.current_fixed_fee_eur - injection_revenue

    @cached_property
    def cost_per_kwh(self) -> float:
        if self.consumption_kwh <= 0:
            return 0.0
        return self.current_cost() / self.consumption_kwh


@dataclass(frozen=True)
class CostComponent:
    """Reference information for explaining electricity invoices."""

    key: str
    label: str
    value_eur_per_mwh: float
    description: str
    adjustable: bool

    @property
    def value_eur_per_kwh(self) -> float:
        return round(self.value_eur_per_mwh / 1000.0, 6)


@dataclass(frozen=True)
class CostGuideSection:
    key: str
    title: str
    subtitle: str
    components: List[CostComponent]
    notes: List[str]

    @property
    def total_eur_per_mwh(self) -> float:
        return round(sum(component.value_eur_per_mwh for component in self.components), 4)

    @property
    def total_eur_per_kwh(self) -> float:
        return round(self.total_eur_per_mwh / 1000.0, 6)


@dataclass
class ShareConstraint:
    min_share: float = 0.0
    max_share: float = 1.0
    override: Optional[float] = None

    def clamp(self, value: float) -> float:
        value = max(self.min_share, min(self.max_share, value))
        return max(0.0, min(1.0, value))


@dataclass
class ScenarioParameters:
    scenario_id: int
    name: str
    community_price_eur_per_kwh: Optional[float]
    price_min_eur_per_kwh: Optional[float]
    price_max_eur_per_kwh: Optional[float]
    price_step_eur_per_kwh: float
    default_share: float
    coverage_cap: float
    community_fixed_fee_total_eur: float
    community_per_member_fee_eur: float
    fee_allocation: str
    community_variable_fee_eur_per_kwh: float
    community_injection_price_eur_per_kwh: Optional[float]
    tariff_context: str
    member_constraints: Dict[int, ShareConstraint] = field(default_factory=dict)

    def share_for_member(self, member_id: int) -> ShareConstraint:
        return self.member_constraints.get(member_id, ShareConstraint(max_share=self.coverage_cap))

    def price_candidates(self) -> List[float]:
        if self.community_price_eur_per_kwh is not None:
            return [self.community_price_eur_per_kwh]
        if self.price_min_eur_per_kwh is None or self.price_max_eur_per_kwh is None:
            return []
        if self.price_step_eur_per_kwh <= 0:
            return []
        prices: List[float] = []
        value = self.price_min_eur_per_kwh
        while value <= self.price_max_eur_per_kwh + 1e-9:
            prices.append(round(value, 6))
            value += self.price_step_eur_per_kwh
        if not prices:
            prices.append(self.price_min_eur_per_kwh)
        return prices


@dataclass
class MemberCostBreakdown:
    member_id: int
    name: str
    share: float
    community_kwh: float
    supplier_kwh: float
    current_cost: float
    community_cost: float
    delta_cost: float
    cost_per_kwh_now: float
    cost_per_kwh_community: float
    community_energy_cost: float
    community_variable_cost: float
    supplier_energy_cost: float
    shared_fee_component: float
    per_member_fee_component: float
    fixed_fee_component: float
    injection_revenue: float
    current_energy_cost: float
    utility: str = ""


@dataclass
class ScenarioEvaluation:
    scenario_id: int
    scenario_name: str
    price_eur_per_kwh: float
    member_breakdowns: List[MemberCostBreakdown]
    total_current_cost: float
    total_community_cost: float
    total_consumption_kwh: float
    total_community_consumption_kwh: float
    total_supplier_consumption_kwh: float

    @property
    def savings(self) -> float:
        return self.total_current_cost - self.total_community_cost

    @property
    def average_cost_per_kwh(self) -> float:
        if self.total_consumption_kwh <= 0:
            return 0.0
        return self.total_community_cost / self.total_consumption_kwh


@dataclass
class OptimizationResult:
    objective: str
    evaluation: ScenarioEvaluation
    shares: Dict[int, float]


FEE_ALLOCATION_ALL_MEMBERS = "all_members"
FEE_ALLOCATION_PARTICIPANTS = "participants"
FEE_ALLOCATION_CONSUMPTION = "consumption"


def reference_cost_guide() -> Dict[str, CostGuideSection]:
    """Return static explanatory data for Stage 3 based on Belgian reference values."""

    traditional_components = [
        CostComponent(
            key="energy",
            label="Énergie (fournisseur)",
            value_eur_per_mwh=130.8,
            description="Achat de l'électricité auprès du fournisseur classique.",
            adjustable=False,
        ),
        CostComponent(
            key="distribution",
            label="Tarif de distribution (DSO)",
            value_eur_per_mwh=101.8,
            description="Utilisation du réseau local basse tension.",
            adjustable=False,
        ),
        CostComponent(
            key="transport",
            label="Tarif de transport (TSO)",
            value_eur_per_mwh=28.1,
            description="Acheminement haute tension par Elia.",
            adjustable=False,
        ),
        CostComponent(
            key="green",
            label="Soutien énergie verte",
            value_eur_per_mwh=28.38,
            description="Certificats verts et obligations régionales.",
            adjustable=False,
        ),
        CostComponent(
            key="connection",
            label="Redevance d'accès",
            value_eur_per_mwh=0.75,
            description="Autres surcharges régulées au kWh.",
            adjustable=False,
        ),
        CostComponent(
            key="excise",
            label="Accise spéciale",
            value_eur_per_mwh=14.21,
            description="Taxe fédérale sur l'électricité.",
            adjustable=False,
        ),
        CostComponent(
            key="levy",
            label="Contribution énergie",
            value_eur_per_mwh=1.92,
            description="Contribution additionnelle régionale.",
            adjustable=False,
        ),
    ]

    community_components = [
        CostComponent(
            key="energy",
            label="Prix interne communauté",
            value_eur_per_mwh=70.0,
            description="Prix cible fixé par la communauté (ajustable).",
            adjustable=True,
        ),
        CostComponent(
            key="distribution",
            label="Tarif distribution (réseau public)",
            value_eur_per_mwh=101.8,
            description="Toujours dû lorsque l'énergie transite par le réseau public.",
            adjustable=False,
        ),
        CostComponent(
            key="transport",
            label="Tarif transport",
            value_eur_per_mwh=28.1,
            description="Part haute tension inchangée.",
            adjustable=False,
        ),
        CostComponent(
            key="green",
            label="Soutien énergie verte",
            value_eur_per_mwh=28.38,
            description="Certificats verts toujours dus.",
            adjustable=False,
        ),
        CostComponent(
            key="connection",
            label="Redevance d'accès",
            value_eur_per_mwh=0.75,
            description="Autres redevances régulées.",
            adjustable=False,
        ),
        CostComponent(
            key="excise",
            label="Accise spéciale",
            value_eur_per_mwh=14.21,
            description="Taxe fédérale identique.",
            adjustable=False,
        ),
        CostComponent(
            key="levy",
            label="Contribution énergie",
            value_eur_per_mwh=1.92,
            description="Contribution régionale.",
            adjustable=False,
        ),
        CostComponent(
            key="community_variable",
            label="Frais variable communauté",
            value_eur_per_mwh=9.0,
            description="Gestion / maintenance mutualisée (ajustable).",
            adjustable=True,
        ),
    ]

    same_site_components = [
        CostComponent(
            key="energy",
            label="Prix interne communauté",
            value_eur_per_mwh=70.0,
            description="Prix interne cible.",
            adjustable=True,
        ),
        CostComponent(
            key="distribution",
            label="Distribution réduite (site unique)",
            value_eur_per_mwh=20.36,
            description="Tarif DSO réduit pour flux internes.",
            adjustable=False,
        ),
        CostComponent(
            key="transport",
            label="Transport réduit",
            value_eur_per_mwh=5.62,
            description="Tarif TSO réduit dans le bâtiment.",
            adjustable=False,
        ),
        CostComponent(
            key="green",
            label="Soutien énergie verte",
            value_eur_per_mwh=28.38,
            description="Obligation certificats verts.",
            adjustable=False,
        ),
        CostComponent(
            key="connection",
            label="Redevance d'accès",
            value_eur_per_mwh=0.75,
            description="Autres redevances.",
            adjustable=False,
        ),
        CostComponent(
            key="excise",
            label="Accise spéciale",
            value_eur_per_mwh=14.21,
            description="Taxe fédérale.",
            adjustable=False,
        ),
        CostComponent(
            key="levy",
            label="Contribution énergie",
            value_eur_per_mwh=1.92,
            description="Contribution régionale.",
            adjustable=False,
        ),
        CostComponent(
            key="community_variable",
            label="Frais variable communauté",
            value_eur_per_mwh=9.0,
            description="Gestion mutualisée.",
            adjustable=True,
        ),
    ]

    guide = {
        "traditional": CostGuideSection(
            key="traditional",
            title="Facture fournisseur classique",
            subtitle="Structure réglementée pour un contrat individuel en Wallonie.",
            components=traditional_components,
            notes=[
                "Les montants sont exprimés en €/MWh (division par 1000 pour obtenir €/kWh).",
                "La TVA (6 % pour les ménages) s'applique sur la somme de tous les éléments variables et fixes.",
                "Les frais fixes annuels (ex. frais fournisseur ou GRD) s'ajoutent en dehors de ces composantes proportionnelles.",
            ],
        ),
        "community_grid": CostGuideSection(
            key="community_grid",
            title="Communauté énergétique (réseau public)",
            subtitle="Les flux passent par le réseau classique : les tarifs régulés restent dus.",
            components=community_components,
            notes=[
                "Le prix interne et les frais de communauté peuvent être ajustés pour équilibrer la répartition des gains.",
                "Les tarifs régulés (DSO, TSO, taxes) ne peuvent pas être optimisés : ils servent de borne basse pour le prix final.",
            ],
        ),
        "community_same_site": CostGuideSection(
            key="community_same_site",
            title="Communauté site unique / même bâtiment",
            subtitle="Exemple de réduction des tarifs réseau lorsque la communauté partage une infrastructure interne.",
            components=same_site_components,
            notes=[
                "Les réductions réseau dépendent du gestionnaire et du type d'installation : à confirmer pour chaque dossier.",
                "Les taxes et surcharges restent dues : elles fixent le minimum réglementaire.",
            ],
        ),
    }

    return guide


def build_member_inputs(members: Iterable) -> List[MemberCostInput]:
    inputs: List[MemberCostInput] = []
    for member in members:
        consumption = _safe_float(
            member.annual_consumption_kwh,
            (member.timeseries_metadata or {}).get("totals", {}).get("consumption_kwh"),
        )
        inputs.append(
            MemberCostInput(
                member_id=member.id,
                name=member.name,
                consumption_kwh=max(consumption, 0.0),
                current_price_eur_per_kwh=_safe_float(member.current_unit_price_eur_per_kwh, 0.0),
                current_fixed_fee_eur=_safe_float(member.current_fixed_fee_eur, 0.0),
                injection_kwh=_safe_float(member.injection_annual_kwh, 0.0),
                injection_price_eur_per_kwh=_safe_float(member.injection_unit_price_eur_per_kwh, 0.0),
                utility=getattr(member, "utility", ""),
            )
        )
    return inputs


def build_scenario_parameters(scenario, member_constraints: Dict[int, ShareConstraint]) -> ScenarioParameters:
    return ScenarioParameters(
        scenario_id=scenario.id,
        name=scenario.name,
        community_price_eur_per_kwh=scenario.community_price_eur_per_kwh,
        price_min_eur_per_kwh=scenario.price_min_eur_per_kwh,
        price_max_eur_per_kwh=scenario.price_max_eur_per_kwh,
        price_step_eur_per_kwh=scenario.price_step_eur_per_kwh,
        default_share=scenario.default_share,
        coverage_cap=scenario.coverage_cap,
        community_fixed_fee_total_eur=scenario.community_fixed_fee_total_eur,
        community_per_member_fee_eur=scenario.community_per_member_fee_eur,
        fee_allocation=scenario.fee_allocation,
        community_variable_fee_eur_per_kwh=scenario.community_variable_fee_eur_per_kwh,
        community_injection_price_eur_per_kwh=scenario.community_injection_price_eur_per_kwh,
        tariff_context=scenario.tariff_context,
        member_constraints=member_constraints,
    )


def evaluate_scenario(
    inputs: List[MemberCostInput],
    params: ScenarioParameters,
    *,
    shares: Optional[Dict[int, float]] = None,
    price: Optional[float] = None,
) -> ScenarioEvaluation:
    price = price if price is not None else _default_price(params)
    if price is None:
        raise ValueError("Aucun prix communautaire défini pour l'évaluation Stage 3")

    member_breakdowns: List[MemberCostBreakdown] = []
    total_current_cost = 0.0
    total_community_cost = 0.0
    total_consumption = 0.0
    total_comm_consumption = 0.0
    total_supplier_consumption = 0.0

    selected_shares: Dict[int, float] = {}

    for member in inputs:
        constraint = params.share_for_member(member.member_id)
        default_share = constraint.override if constraint.override is not None else params.default_share
        share_value = (shares or {}).get(member.member_id, default_share)
        share_value = min(params.coverage_cap, share_value)
        share_value = constraint.clamp(share_value)
        if share_value < constraint.min_share:
            share_value = constraint.min_share
        share_value = max(0.0, min(1.0, share_value))
        selected_shares[member.member_id] = share_value

    participants = [mid for mid, value in selected_shares.items() if value > 1e-6]
    participant_count = len(participants)
    total_consumption = sum(m.consumption_kwh for m in inputs)

    community_kwh_map = {
        m.member_id: selected_shares[m.member_id] * m.consumption_kwh for m in inputs
    }
    total_comm_consumption = sum(community_kwh_map.values())
    supplier_kwh_map = {
        m.member_id: max(m.consumption_kwh - community_kwh_map[m.member_id], 0.0) for m in inputs
    }
    total_supplier_consumption = sum(supplier_kwh_map.values())

    fixed_fee_per_member = 0.0
    if params.fee_allocation == FEE_ALLOCATION_ALL_MEMBERS and inputs:
        fixed_fee_per_member = params.community_fixed_fee_total_eur / len(inputs)
    elif params.fee_allocation == FEE_ALLOCATION_PARTICIPANTS and participant_count:
        fixed_fee_per_member = params.community_fixed_fee_total_eur / participant_count

    proportional_fee_map: Dict[int, float] = {m.member_id: 0.0 for m in inputs}
    if params.fee_allocation == FEE_ALLOCATION_CONSUMPTION and total_comm_consumption > 0:
        for member in inputs:
            share_kwh = community_kwh_map[member.member_id]
            proportional_fee_map[member.member_id] = (
                params.community_fixed_fee_total_eur * (share_kwh / total_comm_consumption)
            )

    for member in inputs:
        share_value = selected_shares[member.member_id]
        community_kwh = community_kwh_map[member.member_id]
        supplier_kwh = supplier_kwh_map[member.member_id]
        current_cost = member.current_cost()
        community_fee_component = proportional_fee_map[member.member_id]
        if params.fee_allocation in (FEE_ALLOCATION_ALL_MEMBERS, FEE_ALLOCATION_PARTICIPANTS):
            community_fee_component = fixed_fee_per_member if share_value > 1e-6 else (
                fixed_fee_per_member if params.fee_allocation == FEE_ALLOCATION_ALL_MEMBERS else 0.0
            )

        per_member_fee_component = (
            params.community_per_member_fee_eur if share_value > 1e-6 else 0.0
        )
        community_energy_cost = community_kwh * price
        community_variable_cost = (
            community_kwh * params.community_variable_fee_eur_per_kwh
        )
        supplier_energy_cost = supplier_kwh * member.current_price_eur_per_kwh
        fixed_fee_component = member.current_fixed_fee_eur
        injection_price = (
            params.community_injection_price_eur_per_kwh
            if params.community_injection_price_eur_per_kwh is not None
            else member.injection_price_eur_per_kwh
        )
        injection_revenue = member.injection_kwh * injection_price

        community_cost = (
            community_energy_cost
            + community_variable_cost
            + supplier_energy_cost
            + fixed_fee_component
            + community_fee_component
            + per_member_fee_component
            - injection_revenue
        )

        cost_per_kwh_now = 0.0 if member.consumption_kwh <= 0 else current_cost / member.consumption_kwh
        cost_per_kwh_comm = 0.0 if member.consumption_kwh <= 0 else community_cost / member.consumption_kwh

        member_breakdowns.append(
            MemberCostBreakdown(
                member_id=member.member_id,
                name=member.name,
                share=share_value,
                community_kwh=community_kwh,
                supplier_kwh=supplier_kwh,
                current_cost=current_cost,
                community_cost=community_cost,
                delta_cost=community_cost - current_cost,
                cost_per_kwh_now=cost_per_kwh_now,
                cost_per_kwh_community=cost_per_kwh_comm,
                community_energy_cost=community_energy_cost,
                community_variable_cost=community_variable_cost,
                supplier_energy_cost=supplier_energy_cost,
                shared_fee_component=community_fee_component,
                per_member_fee_component=per_member_fee_component,
                fixed_fee_component=fixed_fee_component,
                injection_revenue=injection_revenue,
                current_energy_cost=member.consumption_kwh * member.current_price_eur_per_kwh,
                utility=member.utility,
            )
        )
        total_current_cost += current_cost
        total_community_cost += community_cost

    return ScenarioEvaluation(
        scenario_id=params.scenario_id,
        scenario_name=params.name,
        price_eur_per_kwh=price,
        member_breakdowns=member_breakdowns,
        total_current_cost=total_current_cost,
        total_community_cost=total_community_cost,
        total_consumption_kwh=total_consumption,
        total_community_consumption_kwh=total_comm_consumption,
        total_supplier_consumption_kwh=total_supplier_consumption,
    )


def optimize_total_cost(
    inputs: List[MemberCostInput], params: ScenarioParameters
) -> Optional[OptimizationResult]:
    prices = params.price_candidates()
    if not prices:
        price = _default_price(params)
        prices = [price] if price is not None else []
    if not prices:
        return None

    best_result: Optional[OptimizationResult] = None
    for price in prices:
        current_shares = {m.member_id: params.share_for_member(m.member_id).min_share for m in inputs}
        best_eval = evaluate_scenario(inputs, params, shares=current_shares, price=price)
        best_total_cost = best_eval.total_community_cost

        candidates = _potential_savings(inputs, params, price, current_shares)
        for member_id in candidates:
            constraint = params.share_for_member(member_id)
            trial_shares = dict(current_shares)
            trial_shares[member_id] = constraint.max_share
            trial_eval = evaluate_scenario(inputs, params, shares=trial_shares, price=price)
            if trial_eval.total_community_cost < best_total_cost - 1e-6:
                current_shares = trial_shares
                best_eval = trial_eval
                best_total_cost = trial_eval.total_community_cost
        result = OptimizationResult(
            objective="total_cost",
            evaluation=best_eval,
            shares=current_shares,
        )
        if best_result is None or best_eval.total_community_cost < best_result.evaluation.total_community_cost - 1e-6:
            best_result = result
    return best_result


def optimize_average_cost(
    inputs: List[MemberCostInput], params: ScenarioParameters
) -> Optional[OptimizationResult]:
    result = optimize_total_cost(inputs, params)
    if result is None:
        return None
    return OptimizationResult(
        objective="average_cost",
        evaluation=result.evaluation,
        shares=result.shares,
    )


def optimize_member_cost(
    inputs: List[MemberCostInput],
    params: ScenarioParameters,
    member_id: int,
) -> Optional[OptimizationResult]:
    prices = params.price_candidates()
    if not prices:
        price = _default_price(params)
        prices = [price] if price is not None else []
    if not prices:
        return None

    target_input = next((m for m in inputs if m.member_id == member_id), None)
    if target_input is None:
        return None

    best_eval: Optional[ScenarioEvaluation] = None
    best_shares: Dict[int, float] = {}
    for price in prices:
        constraint = params.share_for_member(member_id)
        base_shares = {m.member_id: params.share_for_member(m.member_id).min_share for m in inputs}

        candidate_values = [constraint.min_share, constraint.max_share]
        candidate_values = [value for value in candidate_values if value is not None]
        if not candidate_values:
            candidate_values = [0.0, params.coverage_cap]

        local_best_eval: Optional[ScenarioEvaluation] = None
        local_best_share_value: Optional[float] = None
        for share_value in candidate_values:
            trial_shares = dict(base_shares)
            trial_shares[member_id] = share_value
            eval_result = evaluate_scenario(inputs, params, shares=trial_shares, price=price)
            member_result = next(
                (b for b in eval_result.member_breakdowns if b.member_id == member_id),
                None,
            )
            if member_result is None:
                continue
            if (
                local_best_eval is None
                or member_result.community_cost < _member_cost(local_best_eval, member_id) - 1e-6
            ):
                local_best_eval = eval_result
                local_best_share_value = share_value
        if local_best_eval is None or local_best_share_value is None:
            continue
        if (
            best_eval is None
            or _member_cost(local_best_eval, member_id) < _member_cost(best_eval, member_id) - 1e-6
        ):
            best_eval = local_best_eval
            best_shares = dict(base_shares)
            best_shares[member_id] = local_best_share_value
    if best_eval is None:
        return None
    return OptimizationResult(
        objective=f"member_{member_id}",
        evaluation=best_eval,
        shares=best_shares,
    )


def _potential_savings(
    inputs: List[MemberCostInput],
    params: ScenarioParameters,
    price: float,
    current_shares: Dict[int, float],
) -> List[int]:
    deltas: List[Tuple[float, int]] = []
    baseline_eval = evaluate_scenario(inputs, params, shares=current_shares, price=price)
    baseline_cost = baseline_eval.total_community_cost
    for member in inputs:
        constraint = params.share_for_member(member.member_id)
        if constraint.max_share <= current_shares.get(member.member_id, 0.0) + 1e-9:
            continue
        trial_shares = dict(current_shares)
        trial_shares[member.member_id] = constraint.max_share
        trial_eval = evaluate_scenario(inputs, params, shares=trial_shares, price=price)
        delta = baseline_cost - trial_eval.total_community_cost
        deltas.append((delta, member.member_id))
    deltas.sort(reverse=True)
    return [member_id for delta, member_id in deltas if delta > 1e-6]


def _default_price(params: ScenarioParameters) -> Optional[float]:
    if params.community_price_eur_per_kwh is not None:
        return params.community_price_eur_per_kwh
    if params.price_min_eur_per_kwh is not None and params.price_max_eur_per_kwh is not None:
        return (params.price_min_eur_per_kwh + params.price_max_eur_per_kwh) / 2
    return None


def _member_cost(evaluation: ScenarioEvaluation, member_id: int) -> float:
    breakdown = next((b for b in evaluation.member_breakdowns if b.member_id == member_id), None)
    return breakdown.community_cost if breakdown else float("inf")


def _safe_float(*values) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0
