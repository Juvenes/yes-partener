import os
from datetime import datetime, timedelta
import io
import pandas as pd
from tempfile import NamedTemporaryFile, mkdtemp

from django.core.files.base import ContentFile
from django.test import TestCase, override_settings
from django.urls import reverse

from .forms import StageTwoScenarioForm, StageThreeScenarioForm
from .models import (
    IngestionTemplate,
    Member,
    Profile,
    Project,
    StageTwoScenario,
    StageThreeScenario,
)
from .stage3 import (
    build_member_inputs,
    build_scenario_parameters,
    derive_price_envelope,
    evaluate_scenario,
    generate_trace_rows,
    optimize_everyone_wins,
    optimize_group_benefit,
    price_candidates,
    reference_cost_guide,
)
from .stage2 import load_project_timeseries, build_iteration_configs, evaluate_sharing
from .timeseries import parse_profile_timeseries, build_indexed_template
from . import views


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


class TemplateIndexingTests(TestCase):
    def test_calendar_index_and_leap_year_alignment(self):
        csv_content = """Date+Quart time;consumption;injection;label
2024-06-03 00:00;1;0;Site A
2024-06-03 00:30;2;0;Site A
2024-02-29 00:15;0.5;0.2;Site B
"""

        df = build_indexed_template(ContentFile(csv_content.encode("utf-8")))

        # 00:30 should map to the third quarter of the day.
        row_0030 = df[df["timestamp"].str.contains("00:30")].iloc[0]
        self.assertEqual(row_0030["quarter_index"], 3)
        self.assertEqual(row_0030["week_of_month"], 1)
        self.assertEqual(row_0030["weekday"], 1 + datetime(2024, 6, 3).weekday())

        # Leap-day entry keeps its own calendar slot.
        leap_row = df[df["timestamp"].str.contains("02-29")].iloc[0]
        self.assertEqual(leap_row["month"], "02")
        self.assertEqual(leap_row["week_of_month"], 5)
        self.assertGreater(leap_row["injection_kwh"], 0)

    def test_template_helper_download(self):
        response = self.client.get(reverse("template_helper"))
        self.assertEqual(response.status_code, 200)

        upload = ContentFile(
            b"Date+Quart time,consumption,injection\n2024-01-01 00:00,1,0"
        )
        upload.name = "sample.csv"
        response = self.client.post(reverse("template_helper"), {"timeseries_file": upload})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv")

    def test_template_helper_applies_clean_tags(self):
        upload = ContentFile(
            b"Date+Quart time,consumption,injection\n2024-01-01 00:00,1,0"
        )
        upload.name = "sample.csv"

        response = self.client.post(
            reverse("template_helper"),
            {"timeseries_file": upload, "tags": "Bureau; PV; bureau"},
        )
        self.assertEqual(response.status_code, 200)

        df = pd.read_csv(io.BytesIO(response.content))
        # Duplicate tag is removed, separator normalised
        self.assertEqual(df.iloc[0]["label"], "Bureau, PV")

    @override_settings(MEDIA_ROOT=mkdtemp())
    def test_template_helper_persists_ingestion_template(self):
        upload = ContentFile(
            b"Date+Quart time,consumption,injection\n2024-01-01 00:00,1,0"
        )
        upload.name = "source_timeseries.csv"

        response = self.client.post(
            reverse("template_helper"),
            {"timeseries_file": upload, "tags": "Bureau; PV"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(IngestionTemplate.objects.count(), 1)
        template = IngestionTemplate.objects.first()
        self.assertEqual(template.tags, "Bureau, PV")
        self.assertTrue(template.source_file.name.endswith("source_timeseries.csv"))
        self.assertTrue(template.generated_file.name.endswith("_indexed.csv"))
        self.assertEqual(template.row_count, 1)

    def test_clean_tags_helper(self):
        self.assertEqual(views._clean_tags(""), "")
        self.assertEqual(views._clean_tags("PV;PV;"), "PV")
        self.assertEqual(views._clean_tags("  Maison , PV ; BUREAU"), "Maison, PV, BUREAU")

    @override_settings(MEDIA_ROOT=mkdtemp())
    def test_member_can_reuse_saved_ingestion_template(self):
        project = Project.objects.create(name="Template Project")
        raw = b"Date+Quart time,consumption\n2024-01-01 00:00,2"
        source = ContentFile(raw)
        source.name = "meter.xlsx"

        converted = build_indexed_template(ContentFile(raw))
        buffer = io.StringIO()
        converted.to_csv(buffer, index=False)
        generated = ContentFile(buffer.getvalue().encode("utf-8"))
        generated.name = "meter_indexed.csv"

        template = IngestionTemplate.objects.create(
            name="Modèle compteur",
            tags="Compteur",
            row_count=len(converted.index),
        )
        template.source_file.save(source.name, source, save=False)
        template.generated_file.save(generated.name, generated, save=True)

        response = self.client.post(
            reverse("member_create", args=[project.id]),
            {
                "name": "Membre modèle",
                "data_mode": "timeseries_csv",
                "ingestion_template_id": str(template.id),
                "utility": "",
                "annual_consumption_kwh": "",
                "annual_production_kwh": "",
                "current_unit_price_eur_per_kwh": "0.25",
                "current_fixed_fee_eur": "0",
                "injection_annual_kwh": "0",
                "injection_unit_price_eur_per_kwh": "0.05",
            },
        )

        self.assertRedirects(response, reverse("project_detail", args=[project.id]))
        member = Member.objects.get(project=project, name="Membre modèle")
        self.assertTrue(member.timeseries_file.name.endswith(".csv"))
        self.assertEqual(member.timeseries_metadata["row_count"], template.row_count)


class StageThreeFormsTests(TestCase):
    def test_scenario_form_default_initial(self):
        defaults = StageThreeScenarioForm.default_initial()
        self.assertEqual(defaults["community_price_eur_per_kwh"], "")
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
            community_fixed_fee_total_eur=200,
            community_per_member_fee_eur=50,
        )
        params = build_scenario_parameters(scenario)
        evaluation = evaluate_scenario(
            self._inputs(), params, price=scenario.community_price_eur_per_kwh
        )

        self.assertAlmostEqual(evaluation.total_current_cost, 4920.0, places=2)
        self.assertAlmostEqual(evaluation.total_community_cost, 3720.0, places=2)
        self.assertAlmostEqual(evaluation.savings, 1200.0, places=2)
        breakdown_alpha = next(
            b for b in evaluation.member_breakdowns if b.member_id == self.member_a.id
        )
        self.assertAlmostEqual(breakdown_alpha.share, 1.0)
        self.assertAlmostEqual(breakdown_alpha.community_cost, 2050.0, places=2)
        self.assertAlmostEqual(breakdown_alpha.delta_cost, 550.0, places=2)

    def test_evaluate_scenario_includes_variable_fee_and_injection_override(self):
        scenario = StageThreeScenario.objects.create(
            project=self.project,
            name="Variable",
            community_price_eur_per_kwh=0.20,
            community_variable_fee_eur_per_kwh=0.01,
            community_injection_price_eur_per_kwh=0.06,
        )
        self.member_a.injection_annual_kwh = 500
        self.member_a.injection_unit_price_eur_per_kwh = 0.04
        self.member_a.save()

        params = build_scenario_parameters(scenario)
        evaluation = evaluate_scenario(
            self._inputs(), params, price=scenario.community_price_eur_per_kwh
        )

        breakdown_alpha = next(
            b for b in evaluation.member_breakdowns if b.member_id == self.member_a.id
        )
        self.assertAlmostEqual(breakdown_alpha.community_variable_cost, 100.0, places=2)
        self.assertAlmostEqual(breakdown_alpha.injection_revenue, 30.0, places=2)
        self.assertAlmostEqual(breakdown_alpha.community_cost, 2170.0, places=2)
        self.assertAlmostEqual(breakdown_alpha.delta_cost, 410.0, places=2)

    def test_optimize_group_benefit_uses_derived_price_range(self):
        scenario = StageThreeScenario.objects.create(
            project=self.project,
            name="Optimisation",
        )
        params = build_scenario_parameters(scenario)
        envelope = derive_price_envelope(self._inputs(), params)
        result = optimize_group_benefit(self._inputs(), params, envelope)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.evaluation.price_eur_per_kwh, envelope.effective_min, places=3)
        self.assertGreater(result.evaluation.savings, 0)

    def test_optimize_everyone_wins_respects_member_savings(self):
        scenario = StageThreeScenario.objects.create(
            project=self.project,
            name="Fair",
            price_min_eur_per_kwh=0.25,
            price_max_eur_per_kwh=0.27,
        )
        params = build_scenario_parameters(scenario)
        envelope = derive_price_envelope(self._inputs(), params)
        result = optimize_everyone_wins(self._inputs(), params, envelope)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.evaluation.price_eur_per_kwh, 0.25, places=3)
        for breakdown in result.evaluation.member_breakdowns:
            self.assertGreaterEqual(breakdown.delta_cost, -1e-6)

    def test_generate_trace_rows_matches_evaluation(self):
        scenario = StageThreeScenario.objects.create(
            project=self.project,
            name="Trace",
            community_price_eur_per_kwh=0.18,
        )
        params = build_scenario_parameters(scenario)
        prices = [0.18]
        rows = generate_trace_rows(self._inputs(), params, prices)
        self.assertEqual(len(rows), 2)
        evaluation = evaluate_scenario(self._inputs(), params, price=0.18)
        alpha_row = next(row for row in rows if row["member_id"] == self.member_a.id)
        self.assertAlmostEqual(alpha_row["community_cost_eur"], 1900.0, places=2)
        self.assertAlmostEqual(alpha_row["delta_cost_eur"], 700.0, places=2)
        self.assertAlmostEqual(alpha_row["group_total_savings_eur"], evaluation.savings, places=2)


class StageThreeReferenceTests(TestCase):
    def test_reference_cost_guide_totals(self):
        guide = reference_cost_guide()
        self.assertIn("traditional", guide)
        self.assertIn("community_grid", guide)
        self.assertGreater(guide["traditional"].total_eur_per_kwh, 0)
        self.assertGreater(len(guide["community_grid"].components), 0)


class StageThreeFormTests(TestCase):
    def test_scenario_form_accepts_empty_price(self):
        form = StageThreeScenarioForm(
            data={
                "name": "FormTest",
                "community_price_eur_per_kwh": "",
                "price_min_eur_per_kwh": "",
                "price_max_eur_per_kwh": "",
                "community_fixed_fee_total_eur": "0",
                "community_per_member_fee_eur": "0",
                "community_variable_fee_eur_per_kwh": "0",
                "community_injection_price_eur_per_kwh": "",
                "tariff_context": "community_grid",
                "notes": "",
            }
        )
        self.assertTrue(form.is_valid())

    def test_scenario_form_rejects_inverted_bounds(self):
        form = StageThreeScenarioForm(
            data={
                "name": "Bornes",
                "community_price_eur_per_kwh": "",
                "price_min_eur_per_kwh": "0.30",
                "price_max_eur_per_kwh": "0.20",
                "community_fixed_fee_total_eur": "0",
                "community_per_member_fee_eur": "0",
                "community_variable_fee_eur_per_kwh": "0",
                "community_injection_price_eur_per_kwh": "",
                "tariff_context": "community_grid",
                "notes": "",
            }
        )
        self.assertFalse(form.is_valid())
        self.assertIn("borne minimum", form.errors.as_text())


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
