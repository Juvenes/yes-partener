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
            "name", "utility", "data_mode", "timeseries_file", 
            "annual_consumption_kwh", "annual_production_kwh"
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