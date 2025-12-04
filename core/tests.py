from django.core.files.base import ContentFile
from django.test import TestCase

from .models import Dataset, Member, Project
from .forms import StageThreeTariffForm
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
