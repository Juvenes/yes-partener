from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0010_stagetwoscenario"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="stagethreescenario",
            name="price_step_eur_per_kwh",
        ),
        migrations.RemoveField(
            model_name="stagethreescenario",
            name="default_share",
        ),
        migrations.RemoveField(
            model_name="stagethreescenario",
            name="coverage_cap",
        ),
        migrations.RemoveField(
            model_name="stagethreescenario",
            name="fee_allocation",
        ),
        migrations.DeleteModel(
            name="StageThreeScenarioMember",
        ),
    ]
