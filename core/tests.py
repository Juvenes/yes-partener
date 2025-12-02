from django.core.files.base import ContentFile
from django.test import TestCase
from django.urls import reverse

from .models import Dataset, Member, Project
from .forms import StageThreeScenarioForm
from .stage3 import build_member_inputs
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


class MemberCostInputTests(TestCase):
    def test_member_inputs_use_dataset_totals(self):
        project = Project.objects.create(name="Test")
        dataset = Dataset.objects.create(
            name="Dataset",
            tags=[],
            source_file=ContentFile(b"", name="empty.csv"),
            normalized_file=ContentFile(b"", name="norm.xlsx"),
            metadata={"totals": {"consumption_kwh": 1200}},
        )
        member = Member.objects.create(project=project, name="Alpha", dataset=dataset)
        inputs = build_member_inputs([member])
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].consumption_kwh, 1200)


class StageThreeFormsTests(TestCase):
    def test_scenario_form_default_initial(self):
        defaults = StageThreeScenarioForm.default_initial()
        self.assertIn("community_price_eur_per_kwh", defaults)
