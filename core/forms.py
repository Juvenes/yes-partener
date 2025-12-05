from django import forms
from django.http import QueryDict

from .models import Project, Member, Dataset, StageTwoScenario, Tag


def _normalize_decimal_inputs(data, field_names, prefix=""):
    """Accept both comma and dot decimal separators for numeric fields.

    Browsers configured with a comma decimal separator reject the value before it
    reaches Django's validation. To avoid blocking user input on Stage 3 forms we
    rewrite commas to dots for the relevant fields while keeping other values
    intact.
    """

    if data is None:
        return data

    prefix_str = f"{prefix}-" if prefix else ""
    copied = data.copy() if isinstance(data, QueryDict) else (data.copy() if hasattr(data, "copy") else data)

    for name in field_names:
        key = f"{prefix_str}{name}"
        value = copied.get(key)
        if isinstance(value, str):
            copied[key] = value.replace(",", ".")

    return copied

class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ["name"]

class MemberForm(forms.ModelForm):
    new_tags = forms.CharField(
        required=False,
        help_text="Nouveaux tags séparés par des virgules",
        widget=forms.TextInput(attrs={"placeholder": "bureau, solaire"}),
    )

    class Meta:
        model = Member
        fields = [
            "dataset",
            "name",
            "utility",
            "annual_consumption_kwh",
            "annual_production_kwh",
            "tags",
        ]
        widgets = {
            "tags": forms.SelectMultiple(attrs={"size": 5}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["tags"].queryset = Tag.objects.all()
class DatasetForm(forms.ModelForm):
    tags = forms.ModelMultipleChoiceField(
        queryset=Tag.objects.all(), required=False, widget=forms.SelectMultiple(attrs={"size": 6})
    )
    new_tags = forms.CharField(
        required=False,
        help_text="Nouveaux tags séparés par des virgules",
        widget=forms.TextInput(attrs={"placeholder": "solaire, bureau, 2023"}),
    )

    class Meta:
        model = Dataset
        fields = ["name", "tags", "new_tags", "source_file"]

class StageTwoScenarioForm(forms.ModelForm):
    ITERATION_CHOICES = [
        ("none", "— Ignorer"),
        ("equal", "Clé part égale"),
        ("percentage", "Clé pourcentage fixe"),
        ("proportional", "Clé proportionnelle à la consommation"),
    ]

    def __init__(self, *args, members=None, **kwargs):
        self.members = list(members or [])
        super().__init__(*args, **kwargs)

        for iteration in range(1, 4):
            choice_field = f"iteration_{iteration}_type"
            self.fields[choice_field] = forms.ChoiceField(
                choices=self.ITERATION_CHOICES,
                label=f"Itération {iteration}",
                initial="none",
            )

            for member in self.members:
                percentage_field = f"iteration_{iteration}_member_{member.id}"
                self.fields[percentage_field] = forms.FloatField(
                    label=member.name,
                    required=False,
                    initial=0.0,
                    widget=forms.NumberInput(attrs={"step": "0.01", "min": "0", "max": "100"}),
                )

        if isinstance(self.instance, StageTwoScenario) and self.instance.pk:
            for config in self.instance.iteration_configs():
                choice_field = f"iteration_{config['order']}_type"
                if choice_field in self.fields:
                    self.fields[choice_field].initial = config["key_type"]
                for member_id, value in config.get("percentages", {}).items():
                    field_name = f"iteration_{config['order']}_member_{member_id}"
                    if field_name in self.fields:
                        self.fields[field_name].initial = round(float(value) * 100.0, 6)
        elif not self.initial:
            defaults = self.default_initial()
            for key, value in defaults.items():
                if key in self.fields:
                    self.fields[key].initial = value

    @classmethod
    def default_initial(cls):
        return {"iteration_1_type": "equal"}

    class Meta:
        model = StageTwoScenario
        fields = ["name", "description"]
        widgets = {
            "description": forms.Textarea(attrs={"rows": 2, "placeholder": "Notes ou hypothèses (facultatif)"}),
        }

    def iteration_member_fields(self, iteration):
        return [
            f"iteration_{iteration}_member_{member.id}"
            for member in self.members
            if f"iteration_{iteration}_member_{member.id}" in self.fields
        ]

    def clean(self):
        cleaned = super().clean()
        iterations_payload = []

        for iteration in range(1, 4):
            iteration_type = cleaned.get(f"iteration_{iteration}_type")
            if not iteration_type or iteration_type == "none":
                continue

            config = {"order": iteration, "key_type": iteration_type, "percentages": {}}
            if iteration_type == "percentage":
                percentages = {}
                for member in self.members:
                    field_name = f"iteration_{iteration}_member_{member.id}"
                    value = cleaned.get(field_name)
                    if value is None:
                        value = 0.0
                    if value < 0:
                        self.add_error(field_name, "Les pourcentages doivent être positifs.")
                    percentages[member.id] = max(0.0, float(value))

                total = sum(percentages.values())
                if total <= 0:
                    raise forms.ValidationError(
                        f"L'itération {iteration} doit définir au moins un pourcentage strictement positif."
                    )
                config["percentages"] = {member_id: value / total for member_id, value in percentages.items()}
            iterations_payload.append(config)

        if not iterations_payload:
            raise forms.ValidationError("Définissez au moins une itération de partage active.")

        cleaned["iterations_payload"] = iterations_payload
        return cleaned

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.iterations = self.cleaned_data.get("iterations_payload", [])
        if commit:
            instance.save()
        return instance


class StageThreeTariffForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        data = _normalize_decimal_inputs(kwargs.get("data"), self.Meta.fields, prefix=kwargs.get("prefix") or "")
        if data is not None:
            kwargs["data"] = data

        super().__init__(*args, **kwargs)
        label_map = {
            "supplier_energy_price_eur_per_kwh": "Énergie (fournisseur)",
            "distribution_tariff_eur_per_kwh": "Tarif distribution (DSO)",
            "transport_tariff_eur_per_kwh": "Tarif transport (TSO)",
            "green_support_eur_per_kwh": "Soutien énergie verte",
            "access_fee_eur_per_kwh": "Redevance d'accès",
            "special_excise_eur_per_kwh": "Accise spéciale",
            "energy_contribution_eur_per_kwh": "Contribution énergie",
            "injection_price_eur_per_kwh": "Prix d'injection",
        }
        for key, label in label_map.items():
            if key in self.fields:
                self.fields[key].label = label

    class Meta:
        model = Member
        fields = [
            "supplier_energy_price_eur_per_kwh",
            "distribution_tariff_eur_per_kwh",
            "transport_tariff_eur_per_kwh",
            "green_support_eur_per_kwh",
            "access_fee_eur_per_kwh",
            "special_excise_eur_per_kwh",
            "energy_contribution_eur_per_kwh",
            "injection_price_eur_per_kwh",
        ]
        widgets = {
            field: forms.NumberInput(attrs={"step": "0.000001", "min": "0"})
            for field in fields
        }


class CommunityOptimizationForm(forms.Form):
    COMMUNITY_CHOICES = [
        ("public_grid", "Communauté via réseau public"),
        ("single_building", "Site unique / même bâtiment"),
    ]

    def __init__(self, *args, **kwargs):
        data = _normalize_decimal_inputs(
            kwargs.get("data"),
            [
                "community_fee_eur_per_kwh",
                "reduced_distribution_eur_per_kwh",
                "reduced_transport_eur_per_kwh",
            ],
            prefix=kwargs.get("prefix") or "",
        )
        if data is not None:
            kwargs["data"] = data

        super().__init__(*args, **kwargs)

    community_fee_eur_per_kwh = forms.FloatField(
        label="communityFee (€/kWh)",
        required=False,
        min_value=0.0,
        initial=0.0,
        widget=forms.NumberInput(attrs={"step": "0.000001", "min": "0"}),
    )
    community_type = forms.ChoiceField(
        label="Type de communauté",
        choices=COMMUNITY_CHOICES,
        initial="public_grid",
    )
    reduced_distribution_eur_per_kwh = forms.FloatField(
        label="Distribution réduite (€/kWh)",
        required=False,
        min_value=0.0,
        widget=forms.NumberInput(attrs={"step": "0.000001", "min": "0"}),
    )
    reduced_transport_eur_per_kwh = forms.FloatField(
        label="Transport réduit (€/kWh)",
        required=False,
        min_value=0.0,
        widget=forms.NumberInput(attrs={"step": "0.000001", "min": "0"}),
    )