from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase

import pandas as pd

from .forms import StageThreeTariffForm
from .models import Dataset, Member, Project, Tag
from .stage2 import StageTwoIterationConfig, evaluate_sharing
from .stage3 import build_member_tariffs, compute_baselines
from .timeseries import build_indexed_template, parse_member_timeseries


class DatasetParsingTests(TestCase):
    def test_calendar_index_present(self):
        csv_content = """Date+Quart time;consumption;injection
2024-06-03 00:00;1;0
2024-06-03 00:30;2;0
2024-02-29 00:15;0.5;0.2
"""
        df = build_indexed_template(ContentFile(csv_content.encode("utf-8")))
        self.assertIn("month", df.columns)
        self.assertIn("week_of_month", df.columns)
        self.assertIn("weekday", df.columns)
        self.assertIn("quarter_index", df.columns)

    def test_parse_member_timeseries_normalizes(self):
        upload = ContentFile(b"timestamp,consumption\n2024-01-01 00:00,1")
        upload.name = "sample.csv"
        result = parse_member_timeseries(upload)
        self.assertEqual(result.metadata.row_count, 1)
        self.assertIn("month", result.data.columns)


class MemberTariffTests(TestCase):
    def test_tariff_form_labels(self):
        form = StageThreeTariffForm()
        self.assertEqual(form.fields["supplier_energy_price_eur_per_kwh"].label, "Énergie (fournisseur)")

    def test_baseline_uses_zero_when_missing_timeseries(self):
        project = Project.objects.create(name="Test")
        member = Member.objects.create(project=project, name="Alpha")
        tariffs = build_member_tariffs([member])
        baselines = compute_baselines(None, tariffs)
        self.assertEqual(baselines[member.id].consumption_kwh, 0)


class StageTwoEqualIterationTests(TestCase):
    def test_second_equal_respects_remaining_consumption(self):
        members = [
            type("MemberStub", (), {"id": 1, "name": "A"})(),
            type("MemberStub", (), {"id": 2, "name": "B"})(),
            type("MemberStub", (), {"id": 3, "name": "C"})(),
        ]

        timestamp = pd.to_datetime("2024-01-01 00:00")
        timeseries = pd.DataFrame(
            [
                {"timestamp": timestamp, "member_id": 1, "member_name": "A", "production": 100.0, "consumption": 10.0},
                {"timestamp": timestamp, "member_id": 2, "member_name": "B", "production": 0.0, "consumption": 30.0},
                {"timestamp": timestamp, "member_id": 3, "member_name": "C", "production": 0.0, "consumption": 60.0},
            ]
        )

        iterations = [
            StageTwoIterationConfig(order=1, key_type="equal", percentages={}),
            StageTwoIterationConfig(order=2, key_type="equal", percentages={}),
            StageTwoIterationConfig(order=3, key_type="proportional", percentages={}),
        ]

        evaluation = evaluate_sharing(timeseries, members, iterations)

        allocations = {summary.member_name: summary.community_consumption_kwh for summary in evaluation.member_summaries}

        self.assertAlmostEqual(allocations["A"], 10.0)
        self.assertAlmostEqual(allocations["B"], 30.0)
        self.assertAlmostEqual(allocations["C"], 60.0)


class TaggingTests(TestCase):
    def test_member_and_dataset_tags_attach_and_reuse(self):
        project = Project.objects.create(name="Demo")
        dataset = Dataset.objects.create(
            name="DS",
            source_file=SimpleUploadedFile("src.csv", b"timestamp,consumption\n"),
            normalized_file=SimpleUploadedFile("norm.xlsx", b"binary"),
        )

        solar = Tag.objects.create(name="solaire", color="#aaccee")
        member = Member.objects.create(project=project, name="Alice", dataset=dataset)

        dataset.tags.add(solar)
        new_tag, _ = Tag.objects.get_or_create(name="bureau")
        member.tags.add(solar, new_tag)

        self.assertIn(solar, dataset.tags.all())
        self.assertEqual(member.tags.count(), 2)
        self.assertIn("bureau", list(member.tags.values_list("name", flat=True)))
