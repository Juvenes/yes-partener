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
            "name", "utility", "timeseries_file",
        ]

class MemberProfileForm(forms.ModelForm):
    class Meta:
        model = MemberProfile
        fields = ["profile", "scale_factor"]

class ProfileForm(forms.ModelForm):
    profile_csv = forms.FileField()

    class Meta:
        model = Profile
        fields = ["name", "profile_csv"]

class GlobalParameterForm(forms.ModelForm):
    class Meta:
        model = GlobalParameter
        fields = ["key", "value", "note"]
        widgets = {
            "note": forms.TextInput(attrs={"placeholder": "Description (optional)"}),
        }