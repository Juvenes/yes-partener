"""Helper objects and algorithms powering Stage 3 cost simulations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

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
class ScenarioParameters:
    scenario_id: int
    name: str
    community_price_eur_per_kwh: Optional[float]
    price_min_eur_per_kwh: Optional[float]
    price_max_eur_per_kwh: Optional[float]
    community_fixed_fee_total_eur: float
    community_per_member_fee_eur: float
    community_variable_fee_eur_per_kwh: float
    community_injection_price_eur_per_kwh: Optional[float]
    tariff_context: str


@dataclass
class PriceEnvelope:
    derived_min: Optional[float]
    derived_max: Optional[float]
    floor: Optional[float]
    ceiling: Optional[float]
    effective_min: Optional[float]
    effective_max: Optional[float]

    def is_defined(self) -> bool:
        return (
            self.effective_min is not None
            and self.effective_max is not None
            and self.effective_min <= self.effective_max
        )


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
            description="Gestion mutualisée (ajustable).",
            adjustable=True,
        ),
    ]

    return {
        "traditional": CostGuideSection(
            key="traditional",
            title="Facture fournisseur traditionnelle",
            subtitle="Répartition indicative (référence Wallonie 2025).",
            components=traditional_components,
            notes=[
                "Les valeurs sont indicatives et doivent être adaptées selon le contrat réel.",
                "La TVA à 6 % s'applique sur l'ensemble de ces composantes.",
            ],
        ),
        "community_grid": CostGuideSection(
            key="community_grid",
            title="Communauté via réseau public",
            subtitle="Les tarifs régulés restent dus, seule la composante énergie est ajustable.",
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


def build_member_inputs(members: Iterable) -> List[MemberCostInput]:
    inputs: List[MemberCostInput] = []
    for member in members:
        metadata = {}
        if hasattr(member, "dataset") and member.dataset and member.dataset.metadata:
            metadata = member.dataset.metadata
        totals = metadata.get("totals", {}) if isinstance(metadata, dict) else {}
        consumption = _safe_float(
            member.annual_consumption_kwh,
            totals.get("consumption_kwh"),
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


def build_scenario_parameters(scenario) -> ScenarioParameters:
    return ScenarioParameters(
        scenario_id=scenario.id,
        name=scenario.name,
        community_price_eur_per_kwh=scenario.community_price_eur_per_kwh,
        price_min_eur_per_kwh=scenario.price_min_eur_per_kwh,
        price_max_eur_per_kwh=scenario.price_max_eur_per_kwh,
        community_fixed_fee_total_eur=scenario.community_fixed_fee_total_eur,
        community_per_member_fee_eur=scenario.community_per_member_fee_eur,
        community_variable_fee_eur_per_kwh=scenario.community_variable_fee_eur_per_kwh,
        community_injection_price_eur_per_kwh=scenario.community_injection_price_eur_per_kwh,
        tariff_context=scenario.tariff_context,
    )


def derive_price_envelope(
    inputs: Sequence[MemberCostInput], params: ScenarioParameters
) -> PriceEnvelope:
    valid_costs = [
        member.current_price_eur_per_kwh
        for member in inputs
        if member.consumption_kwh > 0 and member.current_price_eur_per_kwh > 0
    ]

    if not valid_costs:
        return PriceEnvelope(
            derived_min=None,
            derived_max=None,
            floor=params.price_min_eur_per_kwh,
            ceiling=params.price_max_eur_per_kwh,
            effective_min=None,
            effective_max=None,
        )

    derived_min = min(valid_costs)
    derived_max = max(valid_costs)

    effective_min = derived_min
    effective_max = derived_max

    if params.price_min_eur_per_kwh is not None:
        effective_min = max(effective_min, params.price_min_eur_per_kwh)
    if params.price_max_eur_per_kwh is not None:
        effective_max = min(effective_max, params.price_max_eur_per_kwh)

    if effective_min > effective_max:
        effective_min = effective_max = None

    return PriceEnvelope(
        derived_min=derived_min,
        derived_max=derived_max,
        floor=params.price_min_eur_per_kwh,
        ceiling=params.price_max_eur_per_kwh,
        effective_min=effective_min,
        effective_max=effective_max,
    )


def price_candidates(
    envelope: PriceEnvelope,
    *,
    resolution: float = 0.0005,
    max_points: int = 200,
) -> List[float]:
    if not envelope.is_defined():
        return []

    start = envelope.effective_min or 0.0
    end = envelope.effective_max or 0.0
    if end - start < resolution:
        return [round(start, 6)]

    span = end - start
    step = max(resolution, span / max_points)
    prices: List[float] = []
    current = start
    while current <= end + 1e-9:
        prices.append(round(current, 6))
        current += step
    if prices[-1] != round(end, 6):
        prices.append(round(end, 6))
    return prices


def evaluate_scenario(
    inputs: Sequence[MemberCostInput],
    params: ScenarioParameters,
    *,
    price: float,
) -> ScenarioEvaluation:
    if price is None or price < 0:
        raise ValueError("Un prix communautaire valide est requis pour l'évaluation Stage 3")

    member_breakdowns: List[MemberCostBreakdown] = []
    total_current_cost = 0.0
    total_community_cost = 0.0
    total_consumption = 0.0

    participant_count = len(inputs)
    community_fixed_per_member = (
        params.community_fixed_fee_total_eur / participant_count if participant_count else 0.0
    )

    for member in inputs:
        community_kwh = max(member.consumption_kwh, 0.0)
        supplier_kwh = 0.0
        total_consumption += community_kwh

        current_cost = member.current_cost()
        community_energy_cost = community_kwh * price
        community_variable_cost = community_kwh * params.community_variable_fee_eur_per_kwh
        supplier_energy_cost = supplier_kwh * member.current_price_eur_per_kwh
        per_member_fee_component = params.community_per_member_fee_eur
        shared_fee_component = community_fixed_per_member
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
            + shared_fee_component
            + per_member_fee_component
            + fixed_fee_component
            - injection_revenue
        )

        cost_per_kwh_now = 0.0 if member.consumption_kwh <= 0 else current_cost / member.consumption_kwh
        cost_per_kwh_comm = (
            0.0 if member.consumption_kwh <= 0 else community_cost / member.consumption_kwh
        )

        member_breakdowns.append(
            MemberCostBreakdown(
                member_id=member.member_id,
                name=member.name,
                share=1.0,
                community_kwh=community_kwh,
                supplier_kwh=supplier_kwh,
                current_cost=current_cost,
                community_cost=community_cost,
                delta_cost=current_cost - community_cost,
                cost_per_kwh_now=cost_per_kwh_now,
                cost_per_kwh_community=cost_per_kwh_comm,
                community_energy_cost=community_energy_cost,
                community_variable_cost=community_variable_cost,
                supplier_energy_cost=supplier_energy_cost,
                shared_fee_component=shared_fee_component,
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
        total_community_consumption_kwh=total_consumption,
        total_supplier_consumption_kwh=0.0,
    )


def optimize_group_benefit(
    inputs: Sequence[MemberCostInput],
    params: ScenarioParameters,
    envelope: PriceEnvelope,
) -> Optional[OptimizationResult]:
    if params.community_price_eur_per_kwh is not None:
        evaluation = evaluate_scenario(inputs, params, price=params.community_price_eur_per_kwh)
        return OptimizationResult(objective="group_benefit", evaluation=evaluation)

    candidates = price_candidates(envelope)
    if not candidates:
        return None

    best_evaluation: Optional[ScenarioEvaluation] = None
    for candidate in candidates:
        evaluation = evaluate_scenario(inputs, params, price=candidate)
        if best_evaluation is None or evaluation.savings > best_evaluation.savings + 1e-6:
            best_evaluation = evaluation
    if best_evaluation is None:
        return None
    return OptimizationResult(objective="group_benefit", evaluation=best_evaluation)


def optimize_everyone_wins(
    inputs: Sequence[MemberCostInput],
    params: ScenarioParameters,
    envelope: PriceEnvelope,
) -> Optional[OptimizationResult]:
    candidates = []
    if params.community_price_eur_per_kwh is not None:
        candidates = [params.community_price_eur_per_kwh]
    else:
        candidates = price_candidates(envelope)

    if not candidates:
        return None

    feasible: List[ScenarioEvaluation] = []
    for candidate in candidates:
        evaluation = evaluate_scenario(inputs, params, price=candidate)
        if all(b.delta_cost >= -1e-6 for b in evaluation.member_breakdowns):
            feasible.append(evaluation)

    if not feasible:
        return None

    feasible.sort(key=lambda ev: (ev.savings, ev.price_eur_per_kwh))
    best = feasible[-1]
    return OptimizationResult(objective="everyone_wins", evaluation=best)


def generate_trace_rows(
    inputs: Sequence[MemberCostInput],
    params: ScenarioParameters,
    prices: Sequence[float],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for price in prices:
        evaluation = evaluate_scenario(inputs, params, price=price)
        for breakdown in evaluation.member_breakdowns:
            rows.append(
                {
                    "member_id": breakdown.member_id,
                    "member_name": breakdown.name,
                    "community_price_eur_per_kwh": evaluation.price_eur_per_kwh,
                    "consumption_kwh": breakdown.community_kwh,
                    "current_cost_eur": breakdown.current_cost,
                    "community_cost_eur": breakdown.community_cost,
                    "delta_cost_eur": breakdown.delta_cost,
                    "community_energy_cost_eur": breakdown.community_energy_cost,
                    "community_variable_cost_eur": breakdown.community_variable_cost,
                    "community_fixed_fee_component_eur": breakdown.shared_fee_component,
                    "community_per_member_fee_eur": breakdown.per_member_fee_component,
                    "fixed_fee_component_eur": breakdown.fixed_fee_component,
                    "supplier_energy_cost_eur": breakdown.supplier_energy_cost,
                    "injection_revenue_eur": breakdown.injection_revenue,
                    "group_total_current_cost_eur": evaluation.total_current_cost,
                    "group_total_community_cost_eur": evaluation.total_community_cost,
                    "group_total_savings_eur": evaluation.savings,
                    "group_total_consumption_kwh": evaluation.total_consumption_kwh,
                }
            )
    return rows


def _safe_float(*values: Optional[float]) -> float:
    for value in values:
        try:
            if value is None:
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0
