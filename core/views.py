from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_http_methods
from django.http import HttpResponseBadRequest, HttpResponse
from dataclasses import asdict
from typing import Dict, List
from .models import (
    Project,
    Member,
    MemberProfile,
    Profile,
    GlobalParameter,
    StageTwoScenario,
    StageThreeScenario,
    IngestionTemplate,
)
from .forms import (
    ProjectForm,
    MemberForm,
    MemberProfileForm,
    ProfileForm,
    GlobalParameterForm,
    StageTwoScenarioForm,
    StageThreeMemberCostForm,
    StageThreeScenarioForm,
)
from .stage2 import (
    evaluate_sharing as evaluate_stage_two,
    build_iteration_configs,
    load_project_timeseries as load_stage_two_timeseries,
)
from .stage3 import (
    build_member_inputs,
    build_scenario_parameters,
    derive_price_envelope,
    evaluate_scenario,
    generate_trace_rows,
    optimize_group_benefit,
    optimize_everyone_wins,
    price_candidates,
    reference_cost_guide,
)
from .timeseries import (
    TimeseriesError,
    attach_calendar_index,
    build_metadata,
    build_indexed_template,
    parse_member_timeseries,
    parse_profile_timeseries,
)
import csv
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from django.core.files.base import ContentFile
import io
from django.db import IntegrityError # Import IntegrityError
from django.contrib import messages
from django.utils.text import slugify
import re
import pandas as pd


def _parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


INGESTION_TAG_SUGGESTIONS = [
    "Bureau",
    "Ecole",
    "Hôpital",
    "Logements",
    "Maison individuelle",
    "Industrie",
    "PV toiture",
    "Eolien",
    "Chaufferie biomasse",
    "Bornes de recharge",
]


def _clean_tags(raw: str) -> str:
    """Normalise a free-form tags string into a comma-separated label."""

    if not raw:
        return ""

    parts = re.split(r"[;,]", raw)
    cleaned = []
    seen = set()
    for part in parts:
        tag = part.strip()
        if not tag:
            continue
        lower = tag.lower()
        if lower in seen:
            continue
        seen.add(lower)
        cleaned.append(tag)

    return ", ".join(cleaned)


# Simplified project list view
def project_list(request):
    projects = Project.objects.all()
    project_form = ProjectForm()
    return render(
        request,
        "core/project_list.html",
        {
            "projects": projects,
            "project_form": project_form,
        },
    )

# New view for the profiles page
def profile_list(request):
    profiles = Profile.objects.filter(is_active=True).order_by("name")
    profile_form = ProfileForm()
    return render(request, "core/profiles.html", {
        "profiles": profiles,
        "profile_form": profile_form
    })

# New view for the global parameters page
def global_parameter_list(request):
    gp_form = GlobalParameterForm()
    gp_list = GlobalParameter.objects.all()
    return render(request, "core/global_parameters.html", {
        "gp_form": gp_form,
        "gp_list": gp_list
    })


def template_helper(request):
    """Provide a simple template and converter for annual 15-minute files."""

    if request.method == "POST":
        upload = request.FILES.get("timeseries_file")
        raw_tags = request.POST.get("tags", "").strip()
        # Backwards compatible with the old "label" field
        cleaned_tags = _clean_tags(raw_tags)
        label = cleaned_tags or request.POST.get("label", "").strip()
        if not upload:
            messages.error(request, "Veuillez sélectionner un fichier à convertir.")
            return redirect("template_helper")

        try:
            converted = build_indexed_template(upload, label=label or None)
        except TimeseriesError as exc:
            messages.error(request, str(exc))
            return redirect("template_helper")

        buffer = io.StringIO()
        converted.to_csv(buffer, index=False)
        csv_content = buffer.getvalue()
        buffer.seek(0)

        filename = slugify(label or upload.name.rsplit(".", 1)[0] or "timeseries")

        template_record = IngestionTemplate(
            name=label or upload.name,
            tags=cleaned_tags,
            row_count=len(converted.index),
        )

        if hasattr(upload, "seek"):
            upload.seek(0)
        template_record.source_file.save(upload.name, upload, save=False)
        template_record.generated_file.save(
            f"{filename}_indexed.csv", ContentFile(csv_content.encode("utf-8")), save=False
        )
        template_record.save()

        messages.success(
            request,
            "Modèle converti et sauvegardé. Vous pourrez le réutiliser dans la phase 1 des projets.",
        )

        response = HttpResponse(csv_content, content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="{filename}_indexed.csv"'
        return response

    example = build_indexed_template(
        io.StringIO(
            "Date+Quart time;consumption;injection;label\n"
            "2024-06-03 00:00;1;0;Bureau A\n"
            "2024-06-03 00:30;2;0;Bureau A\n"
            "2024-02-29 00:15;0.5;0.2;Parc solaire\n"
        )
    )

    preview = example.head(6).to_dict("records")
    return render(
        request,
        "core/template_helper.html",
        {
            "preview_rows": preview,
            "tag_suggestions": INGESTION_TAG_SUGGESTIONS,
            "ingestion_templates": IngestionTemplate.objects.all(),
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
            "ingestion_templates": IngestionTemplate.objects.all(),
        },
    )

@require_http_methods(["POST"])
def member_create(request, project_id):
    project = get_object_or_404(Project, pk=project_id)
    data_mode = request.POST.get('data_mode')

    if data_mode == 'timeseries_csv':
        form = MemberForm(request.POST, request.FILES)
        if not form.is_valid():
            messages.error(request, f"Formulaire invalide : {form.errors.as_json()}")
            return redirect("project_detail", project_id=project.id)

        upload = form.cleaned_data.get("timeseries_file")
        template = None
        if not upload:
            template_id = request.POST.get("ingestion_template_id")
            if template_id:
                template = get_object_or_404(IngestionTemplate, pk=template_id)
                template.generated_file.open("rb")
                upload = template.generated_file

        if not upload:
            messages.error(
                request,
                "Veuillez sélectionner un fichier de données ou choisir un modèle converti.",
            )
            return redirect("project_detail", project_id=project.id)

        try:
            parse_result = parse_member_timeseries(upload)
        except TimeseriesError as exc:
            messages.error(request, f"Import impossible : {exc}")
            return redirect("project_detail", project_id=project.id)

        metadata = asdict(parse_result.metadata)

        member = Member(
            project=project,
            name=form.cleaned_data["name"],
            utility=form.cleaned_data.get("utility", ""),
            data_mode="timeseries_csv",
            timeseries_metadata=metadata,
        )
        member.save()

        if hasattr(upload, "seek"):
            upload.seek(0)
        filename = getattr(upload, "name", "") or f"member_{member.id}.csv"

        if template:
            template.generated_file.open("rb")
            member.timeseries_file.save(
                filename,
                ContentFile(template.generated_file.read()),
                save=True,
            )
            template.generated_file.close()
        else:
            member.timeseries_file.save(filename, upload, save=True)

        row_count = metadata.get("row_count")
        total_conso = float(metadata.get("totals", {}).get("consumption_kwh", 0.0) or 0.0)
        total_prod = float(metadata.get("totals", {}).get("production_kwh", 0.0) or 0.0)
        messages.success(
            request,
            (
                f"Le membre '{member.name}' a été ajouté. "
                f"{row_count} lignes analysées — Consommation totale : {total_conso:.1f} kWh, "
                f"Production totale : {total_prod:.1f} kWh."
            ),
        )

        if parse_result.metadata.warnings:
            for warning in parse_result.metadata.warnings:
                messages.warning(request, warning)

        return redirect("project_detail", project_id=project.id)

    elif data_mode == 'profile_based':
        profile_ids = request.POST.getlist('profiles')
        raw_annual_consumption = request.POST.get('annual_consumption_kwh')
        raw_annual_production = request.POST.get('annual_production_kwh')

        if not profile_ids:
            messages.error(request, "Veuillez sélectionner au moins un profil.")
            return redirect("project_detail", project_id=project_id)

        def _parse_optional(value, label):
            if value is None:
                return None
            value = value.strip()
            if value == "":
                return None
            try:
                return float(value.replace(",", "."))
            except (TypeError, ValueError):
                messages.error(request, f"Valeur numérique invalide pour {label}.")
                raise

        try:
            annual_prod_value = _parse_optional(raw_annual_production or "", "la production annuelle")
        except Exception:
            return redirect("project_detail", project_id=project_id)

        try:
            annual_cons_value = _parse_optional(raw_annual_consumption or "", "la consommation annuelle")
        except Exception:
            return redirect("project_detail", project_id=project_id)

        profiles = list(Profile.objects.filter(id__in=profile_ids))
        if not profiles:
            messages.error(request, "Aucun profil trouvé pour les identifiants sélectionnés.")
            return redirect("project_detail", project_id=project_id)

        production_series = None
        consumption_series = None
        production_base_total = 0.0
        consumption_base_total = 0.0

        def _resolve_profile_role(profile: Profile) -> tuple[str, bool]:
            """Return the effective role for a profile and whether it required correction."""

            role = (profile.profile_type or "").strip().lower()
            if role not in {"production", "consumption"}:
                role = ""

            totals: dict[str, float] = {}
            if isinstance(profile.metadata, dict):
                totals = profile.metadata.get("totals") or {}

            def _as_float(value) -> float:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0

            prod_total = _as_float(totals.get("production_kwh"))
            cons_total = _as_float(totals.get("consumption_kwh"))

            corrected = False

            if not role:
                if prod_total > cons_total and prod_total > 0:
                    role = "production"
                    corrected = True
                elif cons_total > 0:
                    role = "consumption"
                    corrected = True
            elif role == "production" and prod_total == 0 and cons_total > 0:
                role = "consumption"
                corrected = True
            elif role == "consumption" and cons_total == 0 and prod_total > 0:
                role = "production"
                corrected = True

            if not role:
                role = "consumption"

            return role, corrected

        for profile in profiles:
            effective_role, corrected = _resolve_profile_role(profile)
            profile_df = pd.DataFrame(profile.points)
            if "timestamp" not in profile_df or "value_kwh" not in profile_df:
                messages.error(request, f"Le profil {profile.name} ne contient pas de données exploitables.")
                return redirect("project_detail", project_id=project_id)

            profile_df["timestamp"] = pd.to_datetime(profile_df["timestamp"], errors='coerce')
            profile_df = profile_df.dropna(subset=["timestamp"])
            profile_df = profile_df.sort_values("timestamp")
            profile_series = profile_df.set_index("timestamp")["value_kwh"].astype(float).fillna(0.0)

            if effective_role == "production":
                production_series = profile_series if production_series is None else production_series.add(profile_series, fill_value=0)
                production_base_total += float(profile_series.sum())
            else:
                consumption_series = profile_series if consumption_series is None else consumption_series.add(profile_series, fill_value=0)
                consumption_base_total += float(profile_series.sum())

            if corrected:
                human_role = "production" if effective_role == "production" else "consommation"
                messages.warning(
                    request,
                    (
                        f"Le profil {profile.name} a été traité comme {human_role} car ses totaux ne correspondaient pas "
                        "au type déclaré."
                    ),
                )

        if annual_prod_value is None and production_series is not None and production_base_total > 0:
            annual_prod_value = production_base_total
            formatted_prod = f"{annual_prod_value:,.0f}".replace(",", "\u00a0")
            messages.info(
                request,
                f"Production annuelle non renseignée : utilisation de la somme des profils ({formatted_prod} kWh).",
            )
        if annual_cons_value is None and consumption_series is not None and consumption_base_total > 0:
            annual_cons_value = consumption_base_total
            formatted_cons = f"{annual_cons_value:,.0f}".replace(",", "\u00a0")
            messages.info(
                request,
                f"Consommation annuelle non renseignée : utilisation de la somme des profils ({formatted_cons} kWh).",
            )

        if production_series is None and (annual_prod_value or 0) > 0:
            messages.warning(request, "Aucun profil de production n'a été sélectionné alors qu'une production annuelle est renseignée.")
        if consumption_series is None and (annual_cons_value or 0) > 0:
            messages.warning(request, "Aucun profil de consommation n'a été sélectionné alors qu'une consommation annuelle est renseignée.")

        all_index = None
        if production_series is not None:
            all_index = production_series.index if all_index is None else all_index.union(production_series.index)
        if consumption_series is not None:
            all_index = consumption_series.index if all_index is None else all_index.union(consumption_series.index)

        if all_index is None:
            messages.error(request, "Impossible de générer la série temporelle : aucune donnée de profil valide.")
            return redirect("project_detail", project_id=project_id)

        all_index = all_index.sort_values()
        generated_df = pd.DataFrame(index=all_index)

        if production_series is not None and (annual_prod_value or 0) > 0:
            prod_sum = production_series.sum()
            scale = (annual_prod_value or 0.0) / prod_sum if prod_sum else 0.0
            generated_df["Production"] = production_series.reindex(all_index, fill_value=0) * scale
        else:
            generated_df["Production"] = 0.0

        if consumption_series is not None and (annual_cons_value or 0) > 0:
            cons_sum = consumption_series.sum()
            scale = (annual_cons_value or 0.0) / cons_sum if cons_sum else 0.0
            generated_df["Consommation"] = consumption_series.reindex(all_index, fill_value=0) * scale
        else:
            generated_df["Consommation"] = 0.0

        generated_df = generated_df.fillna(0.0)
        generated_df.reset_index(inplace=True)
        generated_df.rename(columns={"index": "Timestamp"}, inplace=True)

        metadata_input = generated_df.rename(
            columns={
                "Timestamp": "timestamp",
                "Production": "production_kwh",
                "Consommation": "consumption_kwh",
            }
        )
        metadata = asdict(
            build_metadata(
                metadata_input,
                {
                    "timestamp": "timestamp",
                    "production": "production_kwh",
                    "consumption": "consumption_kwh",
                },
                file_type="generated_from_profiles",
            )
        )

        price_default = Member._meta.get_field("current_unit_price_eur_per_kwh").default
        fixed_default = Member._meta.get_field("current_fixed_fee_eur").default
        inj_qty_default = Member._meta.get_field("injection_annual_kwh").default
        inj_price_default = Member._meta.get_field("injection_unit_price_eur_per_kwh").default

        member = Member(
            project=project,
            name=request.POST.get('name'),
            utility=request.POST.get('utility', ''),
            data_mode='timeseries_csv',
            annual_consumption_kwh=annual_cons_value if annual_cons_value is not None else None,
            annual_production_kwh=annual_prod_value if annual_prod_value is not None else None,
            timeseries_metadata=metadata,
            current_unit_price_eur_per_kwh=_parse_float(
                request.POST.get('current_unit_price_eur_per_kwh'), price_default
            ),
            current_fixed_fee_eur=_parse_float(
                request.POST.get('current_fixed_fee_eur'), fixed_default
            ),
            injection_annual_kwh=_parse_float(
                request.POST.get('injection_annual_kwh'), inj_qty_default
            ),
            injection_unit_price_eur_per_kwh=_parse_float(
                request.POST.get('injection_unit_price_eur_per_kwh'), inj_price_default
            ),
        )
        member.save()

        output = io.StringIO()
        generated_df.to_csv(output, index=False)
        csv_file = ContentFile(output.getvalue().encode('utf-8'))
        member.timeseries_file.save(f"generated_{member.id}.csv", csv_file, save=True)

        try:
            regenerated = parse_member_timeseries(member.timeseries_file.path)
            member.timeseries_metadata = asdict(regenerated.metadata)
            totals = regenerated.metadata.totals
            if member.annual_consumption_kwh is None and totals.get('consumption_kwh'):
                member.annual_consumption_kwh = totals.get('consumption_kwh')
            if member.annual_production_kwh is None and totals.get('production_kwh'):
                member.annual_production_kwh = totals.get('production_kwh')
            member.save()
        except TimeseriesError as exc:
            messages.warning(request, f"Le fichier généré n'a pas pu être relu pour vérifier les totaux : {exc}")

        messages.success(
            request,
            f"Le membre '{member.name}' a été créé avec un profil annualisé ({metadata.get('row_count')} points).",
        )

        return redirect("project_detail", project_id=project.id)


    return HttpResponseBadRequest("Mode de données non valide.")



@require_http_methods(["POST"])
def profile_create(request):
    form = ProfileForm(request.POST, request.FILES)
    if form.is_valid():
        upload = form.cleaned_data["profile_csv"]
        profile_type = form.cleaned_data["profile_type"]

        try:
            parse_result = parse_profile_timeseries(upload, profile_type)
        except TimeseriesError as exc:
            messages.error(request, f"Import impossible : {exc}")
            return redirect("profiles")

        df = parse_result.data.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")

        points = [
            {"timestamp": ts.isoformat(), "value_kwh": float(val)}
            for ts, val in zip(df["timestamp"], df["value_kwh"])
        ]

        prof = Profile(
            name=form.cleaned_data["name"],
            profile_type=profile_type,
            points=points,
            metadata=asdict(parse_result.metadata),
        )

        # Build a downsampled daily plot for preview
        daily_series = df.set_index("timestamp")["value_kwh"].resample("D").sum()
        plt.figure(figsize=(12, 4))
        plt.plot(daily_series.index, daily_series.values, label=profile_type.capitalize())
        plt.xlabel('Date')
        plt.ylabel('Énergie (kWh)')
        plt.title(f'Profil {prof.name} – cumul quotidien')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        prof.graph.save(f'{prof.name}.png', ContentFile(buf.read()), save=False)
        buf.close()

        prof.save()
        messages.success(request, f"Profil '{prof.name}' importé ({parse_result.metadata.row_count} points).")
        return redirect("profiles")
    messages.error(request, "Formulaire de profil invalide.")
    return redirect("profiles")

@require_http_methods(["POST"])
def global_parameter_create(request):
    form = GlobalParameterForm(request.POST)
    if form.is_valid():
        form.save()
        return redirect("global_parameters") # Redirect to the new parameters page
    return HttpResponseBadRequest("Invalid parameter")

def csv_template_timeseries(request):
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="timeseries_template.csv"'
    writer = csv.writer(response)
    writer.writerow(["Date+Quart time", "consumption", "injection", "label"])

    base = datetime(2024, 1, 1, 0, 0)
    step = timedelta(minutes=15)
    for _ in range(8):
        writer.writerow([base.strftime("%Y-%m-%d %H:%M"), "", "", "Exemple"])
        base += step
    writer.writerow(["...", "", "", ""])
    writer.writerow(["(35 040 lignes sur une année complète)", "", "", ""])
    return response

def project_analysis_page(request, project_id):
    project = get_object_or_404(Project, pk=project_id)
    members = project.members.filter(data_mode='timeseries_csv').exclude(timeseries_file='')

    if not members.exists():
        messages.warning(request, "Aucun membre avec un fichier de données n'a été trouvé pour l'analyse.")
        return redirect("project_detail", project_id=project.id)

    combined_frames = []
    member_stats = []

    for member in members:
        if not member.timeseries_file:
            continue
        try:
            parse_result = parse_member_timeseries(member.timeseries_file.path)
        except TimeseriesError as exc:
            messages.error(request, f"{member.name} : {exc}")
            continue

        df = parse_result.data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df['member'] = member.name
        combined_frames.append(df)

        metadata_dict = asdict(parse_result.metadata)
        total_production = float(df['production_kwh'].sum())
        total_consumption = float(df['consumption_kwh'].sum())

        totals_section = metadata_dict.setdefault('totals', {})
        totals_section['production_kwh'] = total_production
        totals_section['consumption_kwh'] = total_consumption

        if not metadata_dict.get('start') and not df['timestamp'].empty:
            metadata_dict['start'] = df['timestamp'].min().isoformat()
        if not metadata_dict.get('end') and not df['timestamp'].empty:
            metadata_dict['end'] = df['timestamp'].max().isoformat()
        if not metadata_dict.get('normalized_year') and not df['timestamp'].empty:
            year_mode = df['timestamp'].dt.year.mode()
            if not year_mode.empty:
                metadata_dict['normalized_year'] = int(year_mode.iloc[0])

        member_stats.append({
            'member': member,
            'metadata': metadata_dict,
            'total_production': total_production,
            'total_consumption': total_consumption,
        })

    if not combined_frames:
        messages.error(request, "Impossible de traiter les fichiers fournis.")
        return redirect("project_detail", project_id=project.id)

    combined_df = pd.concat(combined_frames, ignore_index=True)
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
    combined_df = combined_df.dropna(subset=['timestamp'])
    combined_df = combined_df.sort_values('timestamp')

    totals_by_instant = combined_df.groupby('timestamp')[['production_kwh', 'consumption_kwh']].sum()
    totals_by_instant.index = pd.to_datetime(totals_by_instant.index)

    total_production = float(totals_by_instant['production_kwh'].sum())
    total_consumption = float(totals_by_instant['consumption_kwh'].sum())
    net_series = (totals_by_instant['production_kwh'] - totals_by_instant['consumption_kwh']).tolist()

    kpis = {
        'total_production': total_production,
        'total_consumption': total_consumption,
        'net_surplus': float(sum(net_series)),
        'peak_production': float(totals_by_instant['production_kwh'].max()),
        'peak_consumption': float(totals_by_instant['consumption_kwh'].max()),
        'autonomy_coverage': (total_production / total_consumption * 100) if total_consumption > 0 else 0,
    }

    monthly_totals = totals_by_instant.resample('M').sum()
    monthly_categories = [dt.strftime('%Y-%m') for dt in monthly_totals.index]
    monthly_series = {
        'production': monthly_totals['production_kwh'].round(2).tolist(),
        'consumption': monthly_totals['consumption_kwh'].round(2).tolist(),
        'net': (monthly_totals['production_kwh'] - monthly_totals['consumption_kwh']).round(2).tolist(),
    }

    daily_totals = totals_by_instant.resample('D').sum()
    recent_daily = daily_totals.tail(31)
    daily_categories = [dt.strftime('%Y-%m-%d') for dt in recent_daily.index]
    daily_series = {
        'production': recent_daily['production_kwh'].round(2).tolist(),
        'consumption': recent_daily['consumption_kwh'].round(2).tolist(),
    }

    monthly_daily_detail = []
    if not daily_totals.empty:
        grouped = daily_totals.groupby(daily_totals.index.to_period('M'))
        for period, month_df in grouped:
            monthly_daily_detail.append(
                {
                    'label': period.strftime('%Y-%m'),
                    'categories': [dt.strftime('%Y-%m-%d') for dt in month_df.index],
                    'production': month_df['production_kwh'].round(2).tolist(),
                    'consumption': month_df['consumption_kwh'].round(2).tolist(),
                    'net': (month_df['production_kwh'] - month_df['consumption_kwh']).round(2).tolist(),
                }
            )

    avg_profile = totals_by_instant.groupby(totals_by_instant.index.time).mean()
    avg_profile_index = [time.strftime('%H:%M') for time in avg_profile.index]
    avg_profile_series = {
        'production': avg_profile['production_kwh'].round(3).tolist(),
        'consumption': avg_profile['consumption_kwh'].round(3).tolist(),
    }

    total_metadata_df = totals_by_instant.reset_index().rename(
        columns={
            'timestamp': 'timestamp',
            'production_kwh': 'production_kwh',
            'consumption_kwh': 'consumption_kwh',
        }
    )
    aggregate_metadata = asdict(
        build_metadata(
            total_metadata_df,
            {'timestamp': 'timestamp', 'production': 'production_kwh', 'consumption': 'consumption_kwh'},
            file_type='project_aggregate',
        )
    )

    normalized_years = [
        item['metadata'].get('normalized_year')
        for item in member_stats
        if item.get('metadata') and item['metadata'].get('normalized_year') is not None
    ]
    if normalized_years:
        # Prefer the most common year among members for the aggregate summary.
        aggregate_metadata['normalized_year'] = max(set(normalized_years), key=normalized_years.count)

    top_producers = sorted(member_stats, key=lambda x: x['total_production'], reverse=True)[:5]
    top_consumers = sorted(member_stats, key=lambda x: x['total_consumption'], reverse=True)[:5]

    pie_producers = {
        'labels': [item['member'].name for item in top_producers if item['total_production'] > 0],
        'data': [item['total_production'] for item in top_producers if item['total_production'] > 0],
    }
    pie_consumers = {
        'labels': [item['member'].name for item in top_consumers if item['total_consumption'] > 0],
        'data': [item['total_consumption'] for item in top_consumers if item['total_consumption'] > 0],
    }

    context = {
        'project': project,
        'kpis': kpis,
        'member_stats': member_stats,
        'monthly_categories': json.dumps(monthly_categories),
        'monthly_series': json.dumps(monthly_series),
        'daily_categories': json.dumps(daily_categories),
        'daily_series': json.dumps(daily_series),
        'monthly_detail': json.dumps(monthly_daily_detail),
        'avg_profile_categories': json.dumps(avg_profile_index),
        'avg_profile_series': json.dumps(avg_profile_series),
        'aggregate_metadata': aggregate_metadata,
        'pie_producers': json.dumps(pie_producers),
        'pie_consumers': json.dumps(pie_consumers),
    }

    return render(request, "core/project_analysis.html", context)


def project_stage2(request, project_id):
    project = get_object_or_404(Project, pk=project_id)
    all_members = list(project.members.all().order_by("name"))
    members_with_series = [member for member in all_members if member.timeseries_file]
    missing_members = [member for member in all_members if not member.timeseries_file]

    scenario_form = StageTwoScenarioForm(
        request.POST or None,
        members=members_with_series,
        prefix="scenario",
    )
    iteration_blocks = []
    for idx in range(1, 4):
        type_field = scenario_form[f"iteration_{idx}_type"]
        member_field_names = scenario_form.iteration_member_fields(idx)
        member_fields = [scenario_form[name] for name in member_field_names]
        iteration_blocks.append(
            {
                "index": idx,
                "type_field": type_field,
                "member_fields": member_fields,
            }
        )

    if request.method == "POST":
        if not members_with_series:
            messages.error(
                request,
                "Ajoutez d'abord des séries temporelles (Stage 1) avant de configurer le partage (Stage 2).",
            )
        elif scenario_form.is_valid():
            scenario = scenario_form.save(commit=False)
            scenario.project = project
            scenario.save()
            messages.success(request, "Scénario Stage 2 enregistré.")
            return redirect("project_stage2", project_id=project.id)

    timeseries_df = None
    load_warnings: List[str] = []
    load_error = None

    if members_with_series:
        try:
            timeseries_df, load_warnings = load_stage_two_timeseries(members_with_series)
        except ValueError as exc:
            load_error = str(exc)

    scenarios = project.stage2_scenarios.all().order_by("name")
    scenario_cards = []

    key_labels = {
        "equal": "Clé part égale",
        "percentage": "Clé pourcentage fixe",
        "proportional": "Clé proportionnelle conso",
    }

    if timeseries_df is not None and not timeseries_df.empty:
        for scenario in scenarios:
            iteration_payload = build_iteration_configs(
                scenario.iteration_configs(),
                members_with_series,
            )
            try:
                evaluation = evaluate_stage_two(
                    timeseries_df,
                    members_with_series,
                    iteration_payload,
                )
            except ValueError as exc:
                scenario_cards.append(
                    {
                        "scenario": scenario,
                        "error": str(exc),
                        "iterations": [],
                    }
                )
                continue

            preview = evaluation.timeline.head(6)
            if not preview.empty:
                preview_copy = preview.copy()
                preview_copy["timestamp"] = preview_copy["timestamp"].astype(str)
                preview_records = preview_copy.to_dict(orient="records")
            else:
                preview_records = []
            iteration_display = [
                {
                    "order": cfg.order,
                    "label": key_labels.get(cfg.key_type, cfg.key_type),
                    "raw": cfg.key_type,
                }
                for cfg in iteration_payload
            ]
            scenario_cards.append(
                {
                    "scenario": scenario,
                    "evaluation": evaluation,
                    "member_summaries": evaluation.member_summaries,
                    "iteration_stats": evaluation.iteration_stats,
                    "preview": preview_records,
                    "iterations": iteration_display,
                    "warnings": evaluation.warnings,
                }
            )
    elif scenarios and load_error:
        messages.warning(request, load_error)

    context = {
        "project": project,
        "scenario_form": scenario_form,
        "scenario_cards": scenario_cards,
        "members_with_series": members_with_series,
        "missing_members": missing_members,
        "load_warnings": load_warnings,
        "load_error": load_error,
        "iteration_blocks": iteration_blocks,
    }

    return render(request, "core/project_stage2.html", context)


def stage2_scenario_csv(request, scenario_id):
    scenario = get_object_or_404(StageTwoScenario, pk=scenario_id)
    project = scenario.project
    members = list(project.members.all().order_by("name"))
    members_with_series = [member for member in members if member.timeseries_file]

    if not members_with_series:
        messages.error(request, "Aucun membre ne possède de série temporelle exploitable.")
        return redirect("project_stage2", project_id=project.id)

    try:
        timeseries_df, _ = load_stage_two_timeseries(members_with_series)
    except ValueError as exc:
        messages.error(request, str(exc))
        return redirect("project_stage2", project_id=project.id)

    iterations = build_iteration_configs(scenario.iteration_configs(), members_with_series)
    try:
        evaluation = evaluate_stage_two(timeseries_df, members_with_series, iterations)
    except ValueError as exc:
        messages.error(request, str(exc))
        return redirect("project_stage2", project_id=project.id)

    timeline = evaluation.timeline.copy()
    if timeline.empty:
        messages.warning(request, "Aucune donnée à exporter pour ce scénario.")
        return redirect("project_stage2", project_id=project.id)

    if hasattr(timeline["timestamp"], "dt"):
        timeline["timestamp"] = timeline["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        timeline["timestamp"] = timeline["timestamp"].astype(str)

    response = HttpResponse(content_type="text/csv")
    filename = f"stage2_{slugify(project.name)}_{slugify(scenario.name)}.csv"
    response["Content-Disposition"] = f'attachment; filename="{filename}"'

    writer = csv.writer(response)
    writer.writerow(timeline.columns)
    for row in timeline.itertuples(index=False):
        writer.writerow(row)

    return response


def project_stage3(request, project_id):
    project = get_object_or_404(Project, pk=project_id)
    members = list(project.members.all().order_by("name"))
    member_inputs = build_member_inputs(members)
    member_forms = {
        member.id: StageThreeMemberCostForm(instance=member, prefix=f"member-{member.id}")
        for member in members
    }
    member_cards = [
        {
            "input": member_input,
            "form": member_forms.get(member_input.member_id),
            "current_cost": member_input.current_cost(),
            "cost_per_kwh": member_input.cost_per_kwh,
        }
        for member_input in member_inputs
    ]

    guide = reference_cost_guide()
    guide_sections = [
        guide[key]
        for key in ["traditional", "community_grid", "community_same_site"]
        if key in guide
    ]

    scenario_form = StageThreeScenarioForm(
        prefix="scenario-new",
        initial=StageThreeScenarioForm.default_initial(),
    )
    scenarios = project.stage3_scenarios.all().order_by("name")

    scenario_cards = []
    for scenario in scenarios:
        params = build_scenario_parameters(scenario)
        envelope = derive_price_envelope(member_inputs, params)

        evaluation = None
        evaluation_error = None
        try:
            base_price = params.community_price_eur_per_kwh
            if base_price is not None:
                evaluation = evaluate_scenario(member_inputs, params, price=base_price)
        except ValueError as exc:
            evaluation_error = str(exc)

        group_optimization = optimize_group_benefit(member_inputs, params, envelope)
        everyone_optimization = optimize_everyone_wins(member_inputs, params, envelope)

        if evaluation is None and group_optimization is not None:
            evaluation = group_optimization.evaluation

        scenario_cards.append(
            {
                "scenario": scenario,
                "form": StageThreeScenarioForm(
                    instance=scenario, prefix=f"scenario-{scenario.id}"
                ),
                "evaluation": evaluation,
                "evaluation_error": evaluation_error,
                "optimizations": {
                    "group": group_optimization,
                    "everyone": everyone_optimization,
                },
                "price_envelope": envelope,
                "envelope_defined": envelope.is_defined(),
                "params": params,
                "guide": guide.get(scenario.tariff_context),
            }
        )

    total_current_cost = sum(inp.current_cost() for inp in member_inputs)
    total_consumption = sum(inp.consumption_kwh for inp in member_inputs)
    avg_current_cost = total_current_cost / total_consumption if total_consumption > 0 else 0.0

    return render(
        request,
        "core/project_stage3.html",
        {
            "project": project,
            "members": members,
            "member_inputs": member_inputs,
            "member_forms": member_forms,
            "member_cards": member_cards,
            "scenario_form": scenario_form,
            "scenario_cards": scenario_cards,
            "total_current_cost": total_current_cost,
            "total_consumption": total_consumption,
            "avg_current_cost": avg_current_cost,
            "guide_sections": guide_sections,
            "guide_lookup": guide,
        },
    )


@require_http_methods(["POST"])
def stage3_member_update(request, project_id, member_id):
    project = get_object_or_404(Project, pk=project_id)
    member = get_object_or_404(Member, pk=member_id, project=project)
    form = StageThreeMemberCostForm(
        request.POST, instance=member, prefix=f"member-{member.id}"
    )
    if form.is_valid():
        form.save()
        messages.success(request, f"Données tarifaires mises à jour pour {member.name}.")
    else:
        messages.error(request, f"Impossible de mettre à jour {member.name} : {form.errors.as_json()}")
    return redirect("project_stage3", project_id=project.id)


@require_http_methods(["POST"])
def stage3_scenario_create(request, project_id):
    project = get_object_or_404(Project, pk=project_id)
    form = StageThreeScenarioForm(request.POST, prefix="scenario-new")
    if form.is_valid():
        scenario = form.save(commit=False)
        scenario.project = project
        scenario.save()
        messages.success(request, f"Scénario '{scenario.name}' créé.")
    else:
        messages.error(request, f"Création du scénario impossible : {form.errors.as_json()}")
    return redirect("project_stage3", project_id=project.id)


@require_http_methods(["POST"])
def stage3_scenario_update(request, project_id, scenario_id):
    project = get_object_or_404(Project, pk=project_id)
    scenario = get_object_or_404(StageThreeScenario, pk=scenario_id, project=project)
    form = StageThreeScenarioForm(
        request.POST, instance=scenario, prefix=f"scenario-{scenario.id}"
    )
    if form.is_valid():
        form.save()
        messages.success(request, f"Scénario '{scenario.name}' mis à jour.")
    else:
        messages.error(request, f"Mise à jour impossible : {form.errors.as_json()}")
    return redirect("project_stage3", project_id=project.id)


@require_http_methods(["POST"])
def stage3_scenario_delete(request, project_id, scenario_id):
    project = get_object_or_404(Project, pk=project_id)
    scenario = get_object_or_404(StageThreeScenario, pk=scenario_id, project=project)
    scenario.delete()
    messages.success(request, "Scénario supprimé.")
    return redirect("project_stage3", project_id=project.id)


@require_http_methods(["GET"])
def stage3_scenario_trace(request, project_id, scenario_id):
    project = get_object_or_404(Project, pk=project_id)
    scenario = get_object_or_404(StageThreeScenario, pk=scenario_id, project=project)
    members = list(project.members.all().order_by("name"))
    inputs = build_member_inputs(members)
    params = build_scenario_parameters(scenario)
    envelope = derive_price_envelope(inputs, params)

    if params.community_price_eur_per_kwh is not None:
        prices = [params.community_price_eur_per_kwh]
    else:
        prices = price_candidates(envelope)

    if not prices:
        messages.warning(
            request,
            "Aucun prix n'a pu être évalué pour générer la trace Stage 3.",
        )
        return redirect("project_stage3", project_id=project.id)

    rows = generate_trace_rows(inputs, params, prices)
    if not rows:
        messages.warning(request, "Aucune donnée disponible pour l'export Stage 3.")
        return redirect("project_stage3", project_id=project.id)

    response = HttpResponse(content_type="text/csv")
    filename = f"stage3_{slugify(project.name)}_{slugify(scenario.name)}.csv"
    response["Content-Disposition"] = f'attachment; filename="{filename}"'

    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(response, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    return response


