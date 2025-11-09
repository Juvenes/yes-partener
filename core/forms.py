from django import forms
from .models import (
    Project,
    Member,
    MemberProfile,
    Profile,
    GlobalParameter,
    StageThreeScenario,
    StageThreeScenarioMember,
)

class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ["name"]

class MemberForm(forms.ModelForm):
    class Meta:
        model = Member
        fields = [
            "name",
            "utility",
            "data_mode",
            "timeseries_file",
            "annual_consumption_kwh",
            "annual_production_kwh",
            "current_unit_price_eur_per_kwh",
            "current_fixed_fee_eur",
            "injection_annual_kwh",
            "injection_unit_price_eur_per_kwh",
        ]

class MemberProfileForm(forms.ModelForm):
    class Meta:
        model = MemberProfile
        fields = ["profile"]

class ProfileForm(forms.ModelForm):
    profile_csv = forms.FileField()

    class Meta:
        model = Profile
        fields = ["name", "profile_type", "profile_csv"]

class GlobalParameterForm(forms.ModelForm):
    class Meta:
        model = GlobalParameter
        fields = ["key", "value", "note"]
        widgets = {
            "note": forms.TextInput(attrs={"placeholder": "Description (optional)"}),
        }


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
    class Meta:
        model = StageThreeScenario
        fields = [
            "name",
            "community_price_eur_per_kwh",
            "price_min_eur_per_kwh",
            "price_max_eur_per_kwh",
            "price_step_eur_per_kwh",
            "default_share",
            "coverage_cap",
            "community_fixed_fee_total_eur",
            "community_per_member_fee_eur",
            "fee_allocation",
            "notes",
        ]
        widgets = {
            "community_price_eur_per_kwh": forms.NumberInput(attrs={"step": "0.001", "min": "0"}),
            "price_min_eur_per_kwh": forms.NumberInput(attrs={"step": "0.001", "min": "0"}),
            "price_max_eur_per_kwh": forms.NumberInput(attrs={"step": "0.001", "min": "0"}),
            "price_step_eur_per_kwh": forms.NumberInput(attrs={"step": "0.001", "min": "0.0001"}),
            "default_share": forms.NumberInput(attrs={"step": "0.05", "min": "0", "max": "1"}),
            "coverage_cap": forms.NumberInput(attrs={"step": "0.05", "min": "0", "max": "1"}),
            "community_fixed_fee_total_eur": forms.NumberInput(attrs={"step": "0.01", "min": "0"}),
            "community_per_member_fee_eur": forms.NumberInput(attrs={"step": "0.01", "min": "0"}),
            "notes": forms.TextInput(attrs={"placeholder": "Contexte ou hypothèses"}),
        }

    def clean(self):
        cleaned = super().clean()
        community_price = cleaned.get("community_price_eur_per_kwh")
        price_min = cleaned.get("price_min_eur_per_kwh")
        price_max = cleaned.get("price_max_eur_per_kwh")

        if community_price is None and (price_min is None or price_max is None):
            raise forms.ValidationError(
                "Définissez soit un prix communautaire fixe, soit une borne min et max pour l'optimisation."
            )

        if price_min is not None and price_max is not None and price_min > price_max:
            raise forms.ValidationError("La borne minimum doit être inférieure ou égale à la borne maximum.")

        return cleaned


class StageThreeScenarioMemberForm(forms.ModelForm):
    class Meta:
        model = StageThreeScenarioMember
        fields = ["share_override", "min_share", "max_share"]
        widgets = {
            "share_override": forms.NumberInput(attrs={"step": "0.05", "min": "0", "max": "1"}),
            "min_share": forms.NumberInput(attrs={"step": "0.05", "min": "0", "max": "1"}),
            "max_share": forms.NumberInput(attrs={"step": "0.05", "min": "0", "max": "1"}),
        }

    def clean(self):
        cleaned = super().clean()
        min_share = cleaned.get("min_share")
        max_share = cleaned.get("max_share")
        if min_share is not None and max_share is not None and min_share > max_share:
            raise forms.ValidationError("La borne minimale doit être inférieure ou égale à la borne maximale.")
        return cleaned