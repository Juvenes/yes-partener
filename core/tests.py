import os
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile

from django.test import TestCase
from django.urls import reverse

from .forms import StageThreeScenarioForm, StageThreeScenarioMemberForm
from .models import Member, Profile, Project, StageThreeScenario
from .stage3 import (
    ShareConstraint,
    build_member_inputs,
    build_scenario_parameters,
    evaluate_scenario,
    optimize_member_cost,
    optimize_total_cost,
)
from .timeseries import parse_profile_timeseries


class ProfileParsingTests(TestCase):
    def _make_csv(self, header, rows):
        tmp = NamedTemporaryFile(mode="w+", suffix=".csv", delete=False)
        tmp.write(header + "\n")
        for row in rows:
            tmp.write(";".join(row) + "\n")
        tmp.flush()
        return tmp

    def test_consumption_profile_prefers_thousand_column(self):
        tmp = self._make_csv(
            "Date;Conso (1 kWh);Conso 1000kWh",
            [
                ("2022-01-01 00:15", "0.001", "1.0"),
                ("2022-01-01 00:30", "0.002", "2.0"),
                ("2022-01-01 00:45", "0.003", "3.0"),
            ],
        )
        try:
            result = parse_profile_timeseries(tmp.name, "consumption")
        finally:
            tmp.close()
            os.unlink(tmp.name)

        self.assertAlmostEqual(sum(result.data["value_kwh"]), 6.0)
        self.assertEqual(result.metadata.detected_columns["value"], "Conso 1000kWh")
        self.assertAlmostEqual(result.metadata.totals["consumption_kwh"], 6.0)
        self.assertEqual(result.metadata.totals["production_kwh"], 0.0)

    def test_production_profile_prefers_non_thousand_column(self):
        tmp = self._make_csv(
            "Date;Prod (1 kWh);Prod 1000KWh",
            [
                ("2022-01-01 00:15", "0.5", "500"),
                ("2022-01-01 00:30", "0.75", "750"),
            ],
        )
        try:
            result = parse_profile_timeseries(tmp.name, "production")
        finally:
            tmp.close()
            os.unlink(tmp.name)

        self.assertAlmostEqual(sum(result.data["value_kwh"]), 1.25)
        self.assertEqual(result.metadata.detected_columns["value"], "Prod (1 kWh)")
        self.assertAlmostEqual(result.metadata.totals["production_kwh"], 1.25)
        self.assertEqual(result.metadata.totals["consumption_kwh"], 0.0)


class ProfileBasedMemberGenerationTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(name="Profil Project")
        start = datetime(2024, 1, 1, 0, 15)
        points = []
        for i in range(4):
            points.append({
                "timestamp": (start + timedelta(minutes=15 * i)).isoformat(),
                "value_kwh": 0.25,
            })
        self.production_profile = Profile.objects.create(
            name="Profil production test",
            profile_type="production",
            points=points,
            metadata={"row_count": len(points)},
        )

    def test_profile_generation_defaults_to_profile_sum(self):
        url = reverse("member_create", args=[self.project.id])
        response = self.client.post(
            url,
            {
                "name": "Producteur",
                "utility": "Site pilote",
                "data_mode": "profile_based",
                "profiles": [str(self.production_profile.id)],
                "annual_consumption_kwh": "",
                "annual_production_kwh": "",
                "current_unit_price_eur_per_kwh": "0.25",
                "current_fixed_fee_eur": "0",
                "injection_annual_kwh": "0",
                "injection_unit_price_eur_per_kwh": "0.05",
            },
        )
        self.assertRedirects(response, reverse("project_detail", args=[self.project.id]))
        member = Member.objects.get(project=self.project, name="Producteur")
        self.assertTrue(member.timeseries_file.name.endswith(".csv"))
        metadata = member.timeseries_metadata
        self.assertGreater(metadata["totals"]["production_kwh"], 0)
        self.assertAlmostEqual(
            metadata["totals"]["production_kwh"],
            member.annual_production_kwh,
            places=3,
        )


class StageThreeCalculatorTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(name="Stage3 Test Project")
        self.member_a = Member.objects.create(
            project=self.project,
            name="Alpha",
            data_mode="timeseries_csv",
            annual_consumption_kwh=10000,
            current_unit_price_eur_per_kwh=0.25,
            current_fixed_fee_eur=100,
            injection_annual_kwh=0,
            injection_unit_price_eur_per_kwh=0.05,
        )
        self.member_b = Member.objects.create(
            project=self.project,
            name="Bravo",
            data_mode="timeseries_csv",
            annual_consumption_kwh=8000,
            current_unit_price_eur_per_kwh=0.28,
            current_fixed_fee_eur=80,
            injection_annual_kwh=0,
            injection_unit_price_eur_per_kwh=0.05,
        )

    def _inputs(self):
        return build_member_inputs(self.project.members.order_by("name"))

    def test_evaluate_scenario_computes_baseline(self):
        scenario = StageThreeScenario.objects.create(
            project=self.project,
            name="Scénario de test",
            community_price_eur_per_kwh=0.18,
            default_share=1.0,
            coverage_cap=1.0,
            community_fixed_fee_total_eur=200,
            community_per_member_fee_eur=50,
        )
        constraints = {
            self.member_a.id: ShareConstraint(min_share=1.0, max_share=1.0, override=1.0),
            self.member_b.id: ShareConstraint(min_share=0.5, max_share=0.5, override=0.5),
        }
        params = build_scenario_parameters(scenario, constraints)
        evaluation = evaluate_scenario(self._inputs(), params)

        self.assertAlmostEqual(evaluation.total_current_cost, 4920.0, places=2)
        self.assertAlmostEqual(evaluation.total_community_cost, 4120.0, places=2)
        self.assertAlmostEqual(evaluation.savings, 800.0, places=2)
        breakdown_alpha = next(
            b for b in evaluation.member_breakdowns if b.member_id == self.member_a.id
        )
        self.assertAlmostEqual(breakdown_alpha.share, 1.0)
        self.assertAlmostEqual(breakdown_alpha.community_cost, 2050.0, places=2)

    def test_optimize_total_cost_prefers_community_when_cheaper(self):
        scenario = StageThreeScenario.objects.create(
            project=self.project,
            name="Optimisation",
            community_price_eur_per_kwh=0.18,
            default_share=0.0,
            coverage_cap=1.0,
            community_fixed_fee_total_eur=200,
            community_per_member_fee_eur=50,
        )
        params = build_scenario_parameters(scenario, {})
        result = optimize_total_cost(self._inputs(), params)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.evaluation.total_community_cost, 3720.0, places=2)
        self.assertAlmostEqual(result.evaluation.savings, 1200.0, places=2)
        self.assertEqual(result.shares[self.member_a.id], 1.0)
        self.assertEqual(result.shares[self.member_b.id], 1.0)

    def test_optimize_member_cost_targets_single_member(self):
        scenario = StageThreeScenario.objects.create(
            project=self.project,
            name="Solo",
            community_price_eur_per_kwh=0.20,
            default_share=0.0,
            coverage_cap=1.0,
        )
        params = build_scenario_parameters(scenario, {})
        result = optimize_member_cost(self._inputs(), params, self.member_a.id)
        self.assertIsNotNone(result)
        self.assertIn(self.member_a.id, result.shares)
        self.assertGreaterEqual(result.shares[self.member_a.id], 0.0)


class StageThreeFormTests(TestCase):
    def test_scenario_form_requires_price_information(self):
        form = StageThreeScenarioForm(
            data={
                "name": "FormTest",
                "community_price_eur_per_kwh": "",
                "price_min_eur_per_kwh": "",
                "price_max_eur_per_kwh": "",
                "price_step_eur_per_kwh": "0.01",
                "default_share": "1",
                "coverage_cap": "1",
                "community_fixed_fee_total_eur": "0",
                "community_per_member_fee_eur": "0",
                "fee_allocation": "participants",
                "notes": "",
            }
        )
        self.assertFalse(form.is_valid())
        self.assertIn("prix communautaire", form.errors.as_text())

    def test_scenario_member_form_enforces_bounds(self):
        form = StageThreeScenarioMemberForm(
            data={
                "share_override": "0.9",
                "min_share": "0.8",
                "max_share": "0.5",
            }
        )
        self.assertFalse(form.is_valid())
        self.assertIn("borne minimale", form.errors.as_text())
