from django import forms
from .models import (
    Project,
    Member,
    Dataset,
    GlobalParameter,
    StageTwoScenario,
    StageThreeScenario,
)

class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ["name"]

class MemberForm(forms.ModelForm):
    class Meta:
        model = Member
        fields = [
            "dataset",
            "name",
            "utility",
            "annual_consumption_kwh",
            "annual_production_kwh",
            "current_unit_price_eur_per_kwh",
            "current_fixed_fee_eur",
            "injection_annual_kwh",
            "injection_unit_price_eur_per_kwh",
        ]
class DatasetForm(forms.ModelForm):
    tags = forms.CharField(required=False, help_text="Séparez les tags par des virgules")

    class Meta:
        model = Dataset
        fields = ["name", "tags", "source_file"]

class GlobalParameterForm(forms.ModelForm):
    class Meta:
        model = GlobalParameter
        fields = ["key", "value", "note"]
        widgets = {
            "note": forms.TextInput(attrs={"placeholder": "Description (optional)"}),
        }


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


class StageThreeMemberCostForm(forms.ModelForm):
    class Meta:
        model = Member
        fields = [
            "current_unit_price_eur_per_kwh",
            "current_fixed_fee_eur",
            "injection_annual_kwh",
            "injection_unit_price_eur_per_kwh",
        ]
        widgets = {
            "current_unit_price_eur_per_kwh": forms.NumberInput(attrs={"step": "0.001", "min": "0"}),
            "current_fixed_fee_eur": forms.NumberInput(attrs={"step": "0.01", "min": "0"}),
            "injection_annual_kwh": forms.NumberInput(attrs={"step": "0.01", "min": "0"}),
            "injection_unit_price_eur_per_kwh": forms.NumberInput(attrs={"step": "0.001", "min": "0"}),
        }


class StageThreeScenarioForm(forms.ModelForm):
    @classmethod
    def default_initial(cls):
        return {
            "community_price_eur_per_kwh": "",
            "price_min_eur_per_kwh": "",
            "price_max_eur_per_kwh": "",
            "community_variable_fee_eur_per_kwh": 0.009,
            "community_fixed_fee_total_eur": 75.0,
            "community_per_member_fee_eur": 100.0,
            "community_injection_price_eur_per_kwh": 0.05,
            "tariff_context": "community_grid",
        }

    class Meta:
        model = StageThreeScenario
        fields = [
            "name",
            "community_price_eur_per_kwh",
            "price_min_eur_per_kwh",
            "price_max_eur_per_kwh",
            "community_fixed_fee_total_eur",
            "community_per_member_fee_eur",
            "community_variable_fee_eur_per_kwh",
            "community_injection_price_eur_per_kwh",
            "tariff_context",
            "notes",
        ]
        widgets = {
            "community_price_eur_per_kwh": forms.NumberInput(attrs={"step": "0.001", "min": "0"}),
            "price_min_eur_per_kwh": forms.NumberInput(attrs={"step": "0.001", "min": "0"}),
            "price_max_eur_per_kwh": forms.NumberInput(attrs={"step": "0.001", "min": "0"}),
            "community_fixed_fee_total_eur": forms.NumberInput(attrs={"step": "0.01", "min": "0"}),
            "community_per_member_fee_eur": forms.NumberInput(attrs={"step": "0.01", "min": "0"}),
            "community_variable_fee_eur_per_kwh": forms.NumberInput(attrs={"step": "0.001", "min": "0"}),
            "community_injection_price_eur_per_kwh": forms.NumberInput(attrs={"step": "0.001", "min": "0"}),
            "notes": forms.TextInput(attrs={"placeholder": "Contexte ou hypothèses"}),
        }

    def clean(self):
        cleaned = super().clean()
        community_price = cleaned.get("community_price_eur_per_kwh")
        price_min = cleaned.get("price_min_eur_per_kwh")
        price_max = cleaned.get("price_max_eur_per_kwh")

        if price_min is not None and price_max is not None and price_min > price_max:
            raise forms.ValidationError(
                "La borne minimum doit être inférieure ou égale à la borne maximum."
            )

        if community_price is not None:
            if price_min is not None and community_price < price_min:
                self.add_error(
                    "community_price_eur_per_kwh",
                    "Le prix communautaire doit être supérieur ou égal à la borne minimum.",
                )
            if price_max is not None and community_price > price_max:
                self.add_error(
                    "community_price_eur_per_kwh",
                    "Le prix communautaire doit être inférieur ou égal à la borne maximum.",
                )

        return cleaned