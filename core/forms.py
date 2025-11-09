from django import forms
from .models import Project, Member, MemberProfile, Profile, GlobalParameter
from .stage3 import (
    CommunityScenarioParameters,
    OBJECTIVE_CHOICES,
    TariffComponents,
    DEFAULT_TARIFFS,
)

class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ["name"]

class MemberForm(forms.ModelForm):
    class Meta:
        model = Member
        fields = [
            "name", "utility", "data_mode", "timeseries_file",
            "annual_consumption_kwh", "annual_production_kwh",
            "current_unit_price_eur_per_kwh", "current_fixed_annual_fee_eur",
            "injected_energy_kwh", "injection_price_eur_per_kwh",
        ]


class CommunityScenarioForm(forms.Form):
    scenario_name = forms.CharField(
        label="Scenario name",
        max_length=200,
        required=False,
        initial="Scenario A",
    )
    community_price = forms.FloatField(
        label="Community energy price (€/kWh)",
        min_value=0.0,
        initial=0.18,
    )
    enable_price_range = forms.BooleanField(
        label="Explore a price range",
        required=False,
    )
    community_price_min = forms.FloatField(
        label="Minimum community price (€/kWh)",
        min_value=0.0,
        required=False,
    )
    community_price_max = forms.FloatField(
        label="Maximum community price (€/kWh)",
        min_value=0.0,
        required=False,
    )
    price_step = forms.FloatField(
        label="Price step (€/kWh)",
        min_value=0.0001,
        required=False,
    )
    community_fixed_fee_total = forms.FloatField(
        label="Total community fixed fees (€)",
        min_value=0.0,
        required=False,
        initial=0.0,
    )
    community_individual_fee = forms.FloatField(
        label="Per-member community fee (€)",
        min_value=0.0,
        required=False,
        initial=0.0,
    )
    distribution_cost_eur_per_mwh = forms.FloatField(
        label="Distribution (€/MWh)",
        min_value=0.0,
        required=False,
        initial=101.8,
    )
    transport_cost_eur_per_mwh = forms.FloatField(
        label="Transport (€/MWh)",
        min_value=0.0,
        required=False,
        initial=28.1,
    )
    green_surcharge_eur_per_mwh = forms.FloatField(
        label="Green energy surcharge (€/MWh)",
        min_value=0.0,
        required=False,
        initial=28.38,
    )
    connection_fee_eur_per_mwh = forms.FloatField(
        label="Connection (€/MWh)",
        min_value=0.0,
        required=False,
        initial=0.75,
    )
    excise_tax_eur_per_mwh = forms.FloatField(
        label="Excise tax (€/MWh)",
        min_value=0.0,
        required=False,
        initial=14.21,
    )
    federal_tax_eur_per_mwh = forms.FloatField(
        label="Federal contribution (€/MWh)",
        min_value=0.0,
        required=False,
        initial=1.92,
    )
    community_management_fee_eur_per_mwh = forms.FloatField(
        label="Community management (€/MWh)",
        min_value=0.0,
        required=False,
        initial=9.0,
    )
    optimization_objective = forms.ChoiceField(
        label="Optimization objective",
        choices=OBJECTIVE_CHOICES,
        required=False,
        initial=OBJECTIVE_CHOICES[0][0],
    )
    target_member = forms.ModelChoiceField(
        label="Focus on member",
        queryset=Member.objects.none(),
        required=False,
    )

    def __init__(self, members, *args, **kwargs):
        self.members = list(members)
        super().__init__(*args, **kwargs)
        self.fields["target_member"].queryset = Member.objects.filter(id__in=[m.id for m in self.members])

        numeric_attrs = {
            "community_price": {"step": 0.0001, "min": 0},
            "community_individual_fee": {"step": 0.01, "min": 0},
            "community_fixed_fee_total": {"step": 0.01, "min": 0},
            "community_price_min": {"step": 0.0001, "min": 0},
            "community_price_max": {"step": 0.0001, "min": 0},
            "price_step": {"step": 0.0001, "min": 0.0001},
            "distribution_cost_eur_per_mwh": {"step": 0.01, "min": 0},
            "transport_cost_eur_per_mwh": {"step": 0.01, "min": 0},
            "green_surcharge_eur_per_mwh": {"step": 0.01, "min": 0},
            "connection_fee_eur_per_mwh": {"step": 0.01, "min": 0},
            "excise_tax_eur_per_mwh": {"step": 0.01, "min": 0},
            "federal_tax_eur_per_mwh": {"step": 0.01, "min": 0},
            "community_management_fee_eur_per_mwh": {"step": 0.01, "min": 0},
        }
        for field_name, attrs in numeric_attrs.items():
            if field_name in self.fields:
                self.fields[field_name].widget.attrs.update(attrs)

        self.coverage_fields = []
        for member in self.members:
            field_name = f"coverage_{member.id}"
            self.fields[field_name] = forms.FloatField(
                label=f"Community coverage for {member.name} (%)",
                min_value=0.0,
                max_value=100.0,
                required=False,
                initial=100.0,
            )
            self.fields[field_name].widget.attrs.update({"step": 1, "min": 0, "max": 100})
            self.coverage_fields.append((field_name, member))

    def clean(self):
        cleaned_data = super().clean()
        enable_price_range = cleaned_data.get("enable_price_range")
        community_price = cleaned_data.get("community_price")
        price_min = cleaned_data.get("community_price_min")
        price_max = cleaned_data.get("community_price_max")
        price_step = cleaned_data.get("price_step")

        if enable_price_range:
            if price_min is None or price_max is None:
                raise forms.ValidationError("Please provide minimum and maximum prices when enabling the price range.")
            if price_min > price_max:
                raise forms.ValidationError("Minimum community price must be less than or equal to the maximum price.")
            if price_step is None or price_step <= 0:
                raise forms.ValidationError("Please provide a positive price step when exploring a price range.")
        else:
            cleaned_data["community_price_min"] = community_price
            cleaned_data["community_price_max"] = community_price
            cleaned_data["price_step"] = None

        objective = cleaned_data.get("optimization_objective") or OBJECTIVE_CHOICES[0][0]
        target_member = cleaned_data.get("target_member")
        if objective == "minimize_member_cost" and target_member is None:
            raise forms.ValidationError("Please select a member to focus on when minimizing individual cost.")

        for field_name, _ in self.coverage_fields:
            value = cleaned_data.get(field_name)
            if value is None:
                cleaned_data[field_name] = 0.0

        # Ensure tariff fields default to zero when omitted
        tariff_defaults = {
            "distribution_cost_eur_per_mwh": DEFAULT_TARIFFS.distribution_eur_per_kwh * 1000,
            "transport_cost_eur_per_mwh": DEFAULT_TARIFFS.transport_eur_per_kwh * 1000,
            "green_surcharge_eur_per_mwh": DEFAULT_TARIFFS.green_surcharge_eur_per_kwh * 1000,
            "connection_fee_eur_per_mwh": DEFAULT_TARIFFS.connection_eur_per_kwh * 1000,
            "excise_tax_eur_per_mwh": DEFAULT_TARIFFS.excise_tax_eur_per_kwh * 1000,
            "federal_tax_eur_per_mwh": DEFAULT_TARIFFS.federal_tax_eur_per_kwh * 1000,
            "community_management_fee_eur_per_mwh": DEFAULT_TARIFFS.community_management_eur_per_kwh * 1000,
        }
        for field_name, default_value in tariff_defaults.items():
            if cleaned_data.get(field_name) is None:
                cleaned_data[field_name] = default_value

        return cleaned_data

    def build_scenario(self):
        if not self.is_valid():
            raise ValueError("Form must be valid before building the scenario.")

        coverage = {
            member.id: (self.cleaned_data.get(f"coverage_{member.id}") or 0.0) / 100.0
            for member in self.members
        }

        tariffs = TariffComponents.from_per_mwh(
            distribution_eur_per_mwh=self.cleaned_data.get("distribution_cost_eur_per_mwh") or 0.0,
            transport_eur_per_mwh=self.cleaned_data.get("transport_cost_eur_per_mwh") or 0.0,
            green_surcharge_eur_per_mwh=self.cleaned_data.get("green_surcharge_eur_per_mwh") or 0.0,
            connection_eur_per_mwh=self.cleaned_data.get("connection_fee_eur_per_mwh") or 0.0,
            excise_tax_eur_per_mwh=self.cleaned_data.get("excise_tax_eur_per_mwh") or 0.0,
            federal_tax_eur_per_mwh=self.cleaned_data.get("federal_tax_eur_per_mwh") or 0.0,
            community_management_eur_per_mwh=self.cleaned_data.get("community_management_fee_eur_per_mwh") or 0.0,
        )

        return CommunityScenarioParameters(
            name=self.cleaned_data.get("scenario_name") or "Scenario",
            community_price=self.cleaned_data["community_price"],
            allow_price_range=self.cleaned_data.get("enable_price_range", False),
            price_min=self.cleaned_data.get("community_price_min") or self.cleaned_data["community_price"],
            price_max=self.cleaned_data.get("community_price_max") or self.cleaned_data["community_price"],
            price_step=self.cleaned_data.get("price_step"),
            total_community_fee=self.cleaned_data.get("community_fixed_fee_total") or 0.0,
            per_member_fee=self.cleaned_data.get("community_individual_fee") or 0.0,
            coverage_targets=coverage,
            objective=self.cleaned_data.get("optimization_objective") or OBJECTIVE_CHOICES[0][0],
            target_member_id=self.cleaned_data.get("target_member").id if self.cleaned_data.get("target_member") else None,
            tariffs=tariffs,
        )

class MemberProfileForm(forms.ModelForm):
    class Meta:
        model = MemberProfile
        fields = ["profile"]

class ProfileForm(forms.ModelForm):
    profile_csv = forms.FileField()
    dataset_type = forms.ChoiceField(
        choices=(("consumption", "Profil de consommation"), ("production", "Profil de production")),
        initial="consumption",
        label="Type de profil",
    )

    class Meta:
        model = Profile
        fields = ["name", "profile_csv", "dataset_type"]

class GlobalParameterForm(forms.ModelForm):
    class Meta:
        model = GlobalParameter
        fields = ["key", "value", "note"]
        widgets = {
            "note": forms.TextInput(attrs={"placeholder": "Description (optional)"}),
        }