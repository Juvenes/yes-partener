from django import forms
from .models import Project, Member, MemberProfile, Profile, GlobalParameter

class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ["name"]

class MemberForm(forms.ModelForm):
    class Meta:
        model = Member
        fields = [
            "name", "utility", "data_mode",
            "timeseries_file",
            "annual_consumption_kwh", "daily_consumption_kwh",
            "annual_production_kwh", "daily_production_kwh",
        ]

class MemberProfileForm(forms.ModelForm):
    class Meta:
        model = MemberProfile
        fields = ["profile", "scale_factor"]

class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ["name", "kind", "points", "version", "is_active"]
        widgets = {
            "points": forms.Textarea(attrs={"rows": 6, "placeholder": "96 valeurs séparées par des virgules ou JSON"}),
        }
class GlobalParameterForm(forms.ModelForm):
    class Meta:
        model = GlobalParameter
        fields = ["key", "value", "note"]
        widgets = {
            "note": forms.TextInput(attrs={"placeholder": "Description (optionnel)"}),
        }