from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_http_methods
from django.http import HttpResponseBadRequest
from .models import Project, Member, MemberProfile, Profile
from .forms import ProjectForm, MemberForm, MemberProfileForm, ProfileForm, GlobalParameterForm
import csv
from datetime import datetime, timedelta
from django.http import HttpResponse
def project_list(request):
    projects = Project.objects.all()
    project_form = ProjectForm()

    profiles = Profile.objects.filter(is_active=True).order_by("name")
    profile_form = ProfileForm()

    from .models import GlobalParameter
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
    return HttpResponseBadRequest("Projet invalide")

def project_detail(request, project_id):
    project = get_object_or_404(Project, pk=project_id)
    members = project.members.all().prefetch_related("member_profiles__profile")
    member_form = MemberForm()
    member_profile_form = MemberProfileForm()
    profile_form = ProfileForm()
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
            "profile_form": profile_form,
        },
    )

@require_http_methods(["POST"])
def member_create(request, project_id):
    project = get_object_or_404(Project, pk=project_id)
    form = MemberForm(request.POST, request.FILES)
    if form.is_valid():
        m = form.save(commit=False)
        m.project = project
        # léger garde-fou: si mode CSV, exiger un fichier; si profil, fichier optionnel
        if m.data_mode == "timeseries_csv" and not m.timeseries_file:
            return HttpResponseBadRequest("CSV requis pour le mode série 15-min.")
        m.save()
        return redirect("project_detail", project_id=project.id)
    return HttpResponseBadRequest("Membre invalide")

@require_http_methods(["POST"])
def member_profile_add(request, project_id, member_id):
    member = get_object_or_404(Member, pk=member_id, project_id=project_id)
    form = MemberProfileForm(request.POST)
    if form.is_valid():
        mp = form.save(commit=False)
        mp.member = member
        # sécuriser le profil actif
        if not mp.profile.is_active:
            return HttpResponseBadRequest("Profil inactif.")
        mp.save()
        return redirect("project_detail", project_id=project_id)
    return HttpResponseBadRequest("Lien membre-profil invalide")

@require_http_methods(["POST"])
def profile_create(request):
    # Création d'un profil global (points en JSON ou CSV 'a,b,c,...')
    form = ProfileForm(request.POST)
    if form.is_valid():
        prof = form.save(commit=False)
        pts = prof.points
        # accepter "1,2,3" en texte
        if isinstance(pts, str):
            txt = pts.strip()
            if txt.startswith("["):
                import json
                pts = json.loads(txt)
            else:
                pts = [float(x.strip()) for x in txt.split(",") if x.strip()]
            prof.points = pts
        # validations basiques
        if len(prof.points) != 96:
            return HttpResponseBadRequest("Le profil doit contenir 96 valeurs (24h à pas 15 min).")
        prof.save()
        return redirect("project_list")
    return HttpResponseBadRequest("Profil invalide")
@require_http_methods(["POST"])
def profile_create(request):
    form = ProfileForm(request.POST)
    if form.is_valid():
        prof = form.save(commit=False)
        pts = prof.points
        if isinstance(pts, str):
            txt = pts.strip()
            if txt.startswith("["):
                import json
                pts = json.loads(txt)
            else:
                pts = [float(x.strip()) for x in txt.split(",") if x.strip()]
            prof.points = pts
        if len(prof.points) != 96:
            return HttpResponseBadRequest("Le profil doit contenir 96 valeurs (24h à pas 15 min).")
        prof.save()
        return redirect("project_list")
    return HttpResponseBadRequest("Profil invalide")

@require_http_methods(["POST"])
def global_parameter_create(request):
    form = GlobalParameterForm(request.POST)
    if form.is_valid():
        form.save()
        return redirect("project_list")
    return HttpResponseBadRequest("Paramètre invalide")
def csv_template_timeseries(request):
    """CSV modèle: 96 lignes (00:00 -> 23:45), en-têtes Time,Production,Consommation, valeurs vides."""
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="timeseries_template.csv"'
    writer = csv.writer(response)
    writer.writerow(["Time", "Production", "Consommation"])

    t = datetime(2000, 1, 1, 0, 0)  # date arbitraire
    step = timedelta(minutes=15)
    for _ in range(96):
        writer.writerow([t.strftime("%H:%M"), "", ""])
        t += step
    return response