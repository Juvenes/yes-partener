from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_http_methods
from django.http import HttpResponseBadRequest, HttpResponse
from .models import Project, Member, MemberProfile, Profile, GlobalParameter
from .forms import ProjectForm, MemberForm, MemberProfileForm, ProfileForm, GlobalParameterForm
import csv
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from django.core.files.base import ContentFile
import io

def project_list(request):
    projects = Project.objects.all()
    project_form = ProjectForm()

    profiles = Profile.objects.filter(is_active=True).order_by("name")
    profile_form = ProfileForm()

    gp_form = GlobalParameterForm()
    gp_list = GlobalParameter.objects.all()

    return render(
        request,
        "core/project_list.html",
        {
            "projects": projects,
            "project_form": project_form,
            "profiles": profiles,
            "profile_form": profile_form,
            "gp_form": gp_form,
            "gp_list": gp_list,
        },
    )


@require_http_methods(["POST"])
def project_create(request):
    form = ProjectForm(request.POST)
    if form.is_valid():
        p = form.save()
        return redirect("project_detail", project_id=p.id)
    return HttpResponseBadRequest("Invalid project")

def project_detail(request, project_id):
    project = get_object_or_404(Project, pk=project_id)
    members = project.members.all().prefetch_related("member_profiles__profile")
    member_form = MemberForm()
    member_profile_form = MemberProfileForm()
    profiles = Profile.objects.filter(is_active=True).order_by("name")
    return render(
        request,
        "core/project_detail.html",
        {
            "project": project,
            "members": members,
            "member_form": member_form,
            "member_profile_form": member_profile_form,
            "profiles": profiles,
        },
    )

@require_http_methods(["POST"])
def member_create(request, project_id):
    project = get_object_or_404(Project, pk=project_id)
    form = MemberForm(request.POST, request.FILES)
    if form.is_valid():
        m = form.save(commit=False)
        m.project = project
        if not m.timeseries_file:
            return HttpResponseBadRequest("CSV file is required.")
        m.save()
        return redirect("project_detail", project_id=project.id)
    return HttpResponseBadRequest("Invalid member")

@require_http_methods(["POST"])
def member_profile_add(request, project_id, member_id):
    member = get_object_or_404(Member, pk=member_id, project_id=project_id)
    form = MemberProfileForm(request.POST)
    if form.is_valid():
        mp = form.save(commit=False)
        mp.member = member
        if not mp.profile.is_active:
            return HttpResponseBadRequest("Inactive profile.")
        mp.save()
        return redirect("project_detail", project_id=project.id)
    return HttpResponseBadRequest("Invalid member-profile link")

@require_http_methods(["POST"])
def profile_create(request):
    form = ProfileForm(request.POST, request.FILES)
    if form.is_valid():
        profile_csv = request.FILES["profile_csv"]
        decoded_file = profile_csv.read().decode('utf-8').splitlines()
        reader = csv.DictReader(decoded_file)
        points = []
        for row in reader:
            points.append({
                "production": float(row['Production']),
                "consumption": float(row['Consommation'])
            })

        if len(points) != 96:
            return HttpResponseBadRequest("The profile must contain 96 values (24h at 15 min intervals).")

        prof = form.save(commit=False)
        prof.points = points
        
        # Generate and save graph
        production = [p['production'] for p in points]
        consumption = [p['consumption'] for p in points]
        time_intervals = [i*15 for i in range(96)]

        plt.figure(figsize=(10, 5))
        plt.plot(time_intervals, production, label='Production')
        plt.plot(time_intervals, consumption, label='Consumption')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Energy (kWh)')
        plt.title(f'Profile: {prof.name}')
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        prof.graph.save(f'{prof.name}.png', ContentFile(buf.read()), save=False)
        buf.close()

        prof.save()
        return redirect("project_list")
    return HttpResponseBadRequest("Invalid profile")

@require_http_methods(["POST"])
def global_parameter_create(request):
    form = GlobalParameterForm(request.POST)
    if form.is_valid():
        form.save()
        return redirect("project_list")
    return HttpResponseBadRequest("Invalid parameter")

def csv_template_timeseries(request):
    """CSV template: 96 lines (00:00 -> 23:45), headers Time,Production,Consumption, empty values."""
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="timeseries_template.csv"'
    writer = csv.writer(response)
    writer.writerow(["Time", "Production", "Consommation"])

    t = datetime(2000, 1, 1, 0, 0)
    step = timedelta(minutes=15)
    for _ in range(96):
        writer.writerow([t.strftime("%H:%M"), "", ""])
        t += step
    return response