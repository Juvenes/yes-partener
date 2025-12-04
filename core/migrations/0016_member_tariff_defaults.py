from django.core.validators import MinValueValidator
from django.db import migrations, models


OLD_DEFAULTS = {
    "supplier_energy_price_eur_per_kwh": 0.25,
    "distribution_tariff_eur_per_kwh": 0.05,
    "transport_tariff_eur_per_kwh": 0.02,
    "green_support_eur_per_kwh": 0.01,
    "access_fee_eur_per_kwh": 0.0,
    "special_excise_eur_per_kwh": 0.0,
    "energy_contribution_eur_per_kwh": 0.0,
}

NEW_DEFAULTS = {
    "supplier_energy_price_eur_per_kwh": 0.1308,
    "distribution_tariff_eur_per_kwh": 0.1018,
    "transport_tariff_eur_per_kwh": 0.0281,
    "green_support_eur_per_kwh": 0.0284,
    "access_fee_eur_per_kwh": 0.0008,
    "special_excise_eur_per_kwh": 0.0142,
    "energy_contribution_eur_per_kwh": 0.0019,
}


def _is_close(a, b, tol=1e-6):
    return abs((a or 0.0) - b) < tol


def apply_new_defaults(apps, schema_editor):
    Member = apps.get_model("core", "Member")
    for member in Member.objects.all():
        if all(_is_close(getattr(member, field, 0.0) or 0.0, OLD_DEFAULTS[field]) for field in OLD_DEFAULTS):
            for field, value in NEW_DEFAULTS.items():
                setattr(member, field, value)
            member.save(update_fields=list(NEW_DEFAULTS.keys()))


def revert_defaults(apps, schema_editor):
    Member = apps.get_model("core", "Member")
    for member in Member.objects.all():
        if all(_is_close(getattr(member, field, 0.0) or 0.0, NEW_DEFAULTS[field]) for field in NEW_DEFAULTS):
            for field, value in OLD_DEFAULTS.items():
                setattr(member, field, value)
            member.save(update_fields=list(OLD_DEFAULTS.keys()))


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0015_tag_remove_dataset_tags_member_tags_dataset_tags"),
    ]

    operations = [
        migrations.AlterField(
            model_name="member",
            name="supplier_energy_price_eur_per_kwh",
            field=models.FloatField(
                default=0.1308,
                help_text="Prix énergie facturé par le fournisseur (€/kWh).",
                validators=[MinValueValidator(0.0)],
            ),
        ),
        migrations.AlterField(
            model_name="member",
            name="distribution_tariff_eur_per_kwh",
            field=models.FloatField(
                default=0.1018,
                help_text="Tarif de distribution (DSO) (€/kWh).",
                validators=[MinValueValidator(0.0)],
            ),
        ),
        migrations.AlterField(
            model_name="member",
            name="transport_tariff_eur_per_kwh",
            field=models.FloatField(
                default=0.0281,
                help_text="Tarif de transport (TSO) (€/kWh).",
                validators=[MinValueValidator(0.0)],
            ),
        ),
        migrations.AlterField(
            model_name="member",
            name="green_support_eur_per_kwh",
            field=models.FloatField(
                default=0.0284,
                help_text="Soutien énergie verte (€/kWh).",
                validators=[MinValueValidator(0.0)],
            ),
        ),
        migrations.AlterField(
            model_name="member",
            name="access_fee_eur_per_kwh",
            field=models.FloatField(
                default=0.0008,
                help_text="Redevance d'accès (€/kWh).",
                validators=[MinValueValidator(0.0)],
            ),
        ),
        migrations.AlterField(
            model_name="member",
            name="special_excise_eur_per_kwh",
            field=models.FloatField(
                default=0.0142,
                help_text="Accise spéciale (€/kWh).",
                validators=[MinValueValidator(0.0)],
            ),
        ),
        migrations.AlterField(
            model_name="member",
            name="energy_contribution_eur_per_kwh",
            field=models.FloatField(
                default=0.0019,
                help_text="Contribution énergie (€/kWh).",
                validators=[MinValueValidator(0.0)],
            ),
        ),
        migrations.RunPython(apply_new_defaults, revert_defaults),
    ]
