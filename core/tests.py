import os
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile

from django.core.files.base import ContentFile
from django.test import TestCase
from django.urls import reverse

from .forms import StageTwoScenarioForm, StageThreeScenarioForm, StageThreeScenarioMemberForm
from .models import Member, Profile, Project, StageTwoScenario, StageThreeScenario
from .stage3 import (
    ShareConstraint,
    build_member_inputs,
    build_scenario_parameters,
    evaluate_scenario,
    optimize_member_cost,
    optimize_total_cost,
    reference_cost_guide,
)
from .stage2 import load_project_timeseries, build_iteration_configs, evaluate_sharing
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

    def test_profile_generation_recovers_from_mislabelled_type(self):
        mislabelled = Profile.objects.create(
            name="Profil production mal typé",
            profile_type="consumption",
            points=self.production_profile.points,
            metadata={
                "row_count": len(self.production_profile.points),
                "totals": {
                    "production_kwh": sum(point["value_kwh"] for point in self.production_profile.points),
                    "consumption_kwh": 0.0,
                },
            },
        )

        url = reverse("member_create", args=[self.project.id])
        response = self.client.post(
            url,
            {
                "name": "Producteur corrigé",
                "utility": "Site pilote",
                "data_mode": "profile_based",
                "profiles": [str(mislabelled.id)],
                "annual_consumption_kwh": "",
                "annual_production_kwh": "",
                "current_unit_price_eur_per_kwh": "0.25",
                "current_fixed_fee_eur": "0",
                "injection_annual_kwh": "0",
                "injection_unit_price_eur_per_kwh": "0.05",
            },
        )
        self.assertRedirects(response, reverse("project_detail", args=[self.project.id]))

        member = Member.objects.get(project=self.project, name="Producteur corrigé")
        metadata = member.timeseries_metadata
        self.assertGreater(metadata["totals"]["production_kwh"], 0)
        self.assertAlmostEqual(
            metadata["totals"]["consumption_kwh"],
            0.0,
            places=5,
        )


class StageThreeFormsTests(TestCase):
    def test_scenario_form_default_initial(self):
        defaults = StageThreeScenarioForm.default_initial()
        self.assertAlmostEqual(defaults["community_price_eur_per_kwh"], 0.07)
        self.assertEqual(defaults["tariff_context"], "community_grid")
        self.assertGreater(defaults["community_variable_fee_eur_per_kwh"], 0)


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

    def test_evaluate_scenario_includes_variable_fee_and_injection_override(self):
        scenario = StageThreeScenario.objects.create(
            project=self.project,
            name="Variable",
            community_price_eur_per_kwh=0.20,
            default_share=0.0,
            coverage_cap=1.0,
            community_variable_fee_eur_per_kwh=0.01,
            community_injection_price_eur_per_kwh=0.06,
        )
        self.member_a.injection_annual_kwh = 500
        self.member_a.injection_unit_price_eur_per_kwh = 0.04
        self.member_a.save()

        constraints = {
            self.member_a.id: ShareConstraint(min_share=1.0, max_share=1.0, override=1.0),
            self.member_b.id: ShareConstraint(min_share=0.0, max_share=0.0, override=0.0),
        }
        params = build_scenario_parameters(scenario, constraints)
        evaluation = evaluate_scenario(self._inputs(), params)

        breakdown_alpha = next(
            b for b in evaluation.member_breakdowns if b.member_id == self.member_a.id
        )
        self.assertAlmostEqual(breakdown_alpha.community_variable_cost, 100.0, places=2)
        self.assertAlmostEqual(breakdown_alpha.injection_revenue, 30.0, places=2)
        self.assertAlmostEqual(breakdown_alpha.community_cost, 2170.0, places=2)

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


class StageThreeReferenceTests(TestCase):
    def test_reference_cost_guide_totals(self):
        guide = reference_cost_guide()
        self.assertIn("traditional", guide)
        self.assertIn("community_grid", guide)
        self.assertGreater(guide["traditional"].total_eur_per_kwh, 0)
        self.assertGreater(len(guide["community_grid"].components), 0)


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


class StageTwoScenarioFormTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(name="Stage2 Form Project")
        self.member_a = Member.objects.create(project=self.project, name="Alpha")
        self.member_b = Member.objects.create(project=self.project, name="Bravo")

    def test_percentage_iteration_normalises_values(self):
        form = StageTwoScenarioForm(
            data={
                "name": "Scénario test",
                "description": "Démo",
                "iteration_1_type": "percentage",
                f"iteration_1_member_{self.member_a.id}": "60",
                f"iteration_1_member_{self.member_b.id}": "40",
                "iteration_2_type": "equal",
                "iteration_3_type": "none",
            },
            members=[self.member_a, self.member_b],
        )

        self.assertTrue(form.is_valid())
        payload = form.cleaned_data["iterations_payload"]
        self.assertEqual(len(payload), 2)
        first = payload[0]
        self.assertEqual(first["key_type"], "percentage")
        self.assertAlmostEqual(first["percentages"][self.member_a.id], 0.6)
        self.assertAlmostEqual(first["percentages"][self.member_b.id], 0.4)
        second = payload[1]
        self.assertEqual(second["key_type"], "equal")

        scenario = form.save(commit=False)
        scenario.project = self.project
        scenario.save()
        self.assertEqual(len(scenario.iterations), 2)


class StageTwoEvaluationTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(name="Stage2 Eval Project")
        self.producer = Member.objects.create(project=self.project, name="Producteur")
        self.consumer = Member.objects.create(project=self.project, name="Consommateur")

        producer_csv = "Time,Production,Consommation\n2024-01-01 00:15,6,1\n2024-01-01 00:30,4,1\n"
        consumer_csv = "Time,Production,Consommation\n2024-01-01 00:15,0,4\n2024-01-01 00:30,0,2\n"
        self.producer.timeseries_file.save("producer.csv", ContentFile(producer_csv), save=True)
        self.consumer.timeseries_file.save("consumer.csv", ContentFile(consumer_csv), save=True)

    def test_equal_then_proportional_allocation(self):
        scenario = StageTwoScenario.objects.create(
            project=self.project,
            name="Partage test",
            iterations=[
                {"order": 1, "key_type": "equal", "percentages": {}},
                {"order": 2, "key_type": "proportional", "percentages": {}},
            ],
        )

        timeseries_df, warnings = load_project_timeseries([self.producer, self.consumer])
        self.assertFalse(timeseries_df.empty)
        self.assertEqual(warnings, [])

        configs = build_iteration_configs(scenario.iteration_configs(), [self.producer, self.consumer])
        evaluation = evaluate_sharing(timeseries_df, [self.producer, self.consumer], configs)

        self.assertAlmostEqual(evaluation.total_community_allocation_kwh, 8.0)
        self.assertAlmostEqual(evaluation.total_remaining_production_kwh, 2.0)
        self.assertAlmostEqual(evaluation.total_unserved_consumption_kwh, 0.0)
        self.assertEqual(len(evaluation.warnings), 2)
        self.assertIn("Itération 2", evaluation.warnings[0])

        summaries = {summary.member_name: summary for summary in evaluation.member_summaries}
        producer_summary = summaries["Producteur"]
        consumer_summary = summaries["Consommateur"]

        self.assertAlmostEqual(producer_summary.total_production_kwh, 10.0)
        self.assertAlmostEqual(producer_summary.shared_production_kwh, 8.0)
        self.assertAlmostEqual(producer_summary.unused_production_kwh, 2.0)
        self.assertAlmostEqual(producer_summary.community_consumption_kwh, 2.0)
        self.assertAlmostEqual(producer_summary.external_consumption_kwh, 0.0)

        self.assertAlmostEqual(consumer_summary.total_consumption_kwh, 6.0)
        self.assertAlmostEqual(consumer_summary.community_consumption_kwh, 6.0)
        self.assertAlmostEqual(consumer_summary.external_consumption_kwh, 0.0)
        self.assertAlmostEqual(consumer_summary.total_production_kwh, 0.0)
