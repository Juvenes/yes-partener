from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_http_methods
from django.http import HttpResponseBadRequest, HttpResponse
from dataclasses import asdict
from .models import Project, Member, MemberProfile, Profile, GlobalParameter
from .forms import ProjectForm, MemberForm, MemberProfileForm, ProfileForm, GlobalParameterForm
from .timeseries import (
    TimeseriesError,
    build_metadata,
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
import pandas as pd


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
    data_mode = request.POST.get('data_mode')

    if data_mode == 'timeseries_csv':
        form = MemberForm(request.POST, request.FILES)
        if not form.is_valid():
            messages.error(request, f"Formulaire invalide : {form.errors.as_json()}")
            return redirect("project_detail", project_id=project.id)

        upload = form.cleaned_data.get("timeseries_file")
        if not upload:
            messages.error(request, "Veuillez sélectionner un fichier de données.")
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

        upload.seek(0)
        filename = upload.name or f"member_{member.id}.csv"
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
        annual_consumption = request.POST.get('annual_consumption_kwh')
        annual_production = request.POST.get('annual_production_kwh')

        if not profile_ids:
            messages.error(request, "Veuillez sélectionner au moins un profil.")
            return redirect("project_detail", project_id=project_id)
        if not (annual_consumption or annual_production):
            messages.error(request, "Veuillez fournir une consommation ou production annuelle.")
            return redirect("project_detail", project_id=project_id)

        try:
            annual_prod_value = float(annual_production) if annual_production else 0.0
            annual_cons_value = float(annual_consumption) if annual_consumption else 0.0

            profiles = list(Profile.objects.filter(id__in=profile_ids))
            if not profiles:
                messages.error(request, "Aucun profil trouvé pour les identifiants sélectionnés.")
                return redirect("project_detail", project_id=project_id)

            production_series = None
            consumption_series = None

            for profile in profiles:
                profile_df = pd.DataFrame(profile.points)
                if "timestamp" not in profile_df or "value_kwh" not in profile_df:
                    messages.error(request, f"Le profil {profile.name} ne contient pas de données exploitables.")
                    return redirect("project_detail", project_id=project_id)

                profile_df["timestamp"] = pd.to_datetime(profile_df["timestamp"], errors='coerce')
                profile_df = profile_df.dropna(subset=["timestamp"])
                profile_df = profile_df.sort_values("timestamp")
                profile_df = profile_df.set_index("timestamp")["value_kwh"]

                if profile.profile_type == "production":
                    production_series = profile_df if production_series is None else production_series.add(profile_df, fill_value=0)
                else:
                    consumption_series = profile_df if consumption_series is None else consumption_series.add(profile_df, fill_value=0)

            if annual_prod_value > 0 and production_series is None:
                messages.warning(request, "Aucun profil de production n'a été sélectionné alors qu'une production annuelle est renseignée.")
            if annual_cons_value > 0 and consumption_series is None:
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

            if production_series is not None and annual_prod_value > 0:
                prod_sum = production_series.sum()
                scale = annual_prod_value / prod_sum if prod_sum else 0.0
                generated_df["Production"] = production_series.reindex(all_index, fill_value=0) * scale
            else:
                generated_df["Production"] = 0.0

            if consumption_series is not None and annual_cons_value > 0:
                cons_sum = consumption_series.sum()
                scale = annual_cons_value / cons_sum if cons_sum else 0.0
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

            member = Member(
                project=project,
                name=request.POST.get('name'),
                utility=request.POST.get('utility', ''),
                data_mode='timeseries_csv',
                annual_consumption_kwh=(annual_cons_value if annual_consumption else None),
                annual_production_kwh=(annual_prod_value if annual_production else None),
                timeseries_metadata=metadata,
            )
            member.save()

            output = io.StringIO()
            generated_df.to_csv(output, index=False)
            csv_file = ContentFile(output.getvalue().encode('utf-8'))
            member.timeseries_file.save(f"generated_{member.id}.csv", csv_file, save=True)

            messages.success(request, f"Le membre '{member.name}' a été créé avec un profil annualisé ({metadata.get('row_count')} points).")

        except Exception as e:
            messages.error(request, f"Une erreur est survenue : {e}")

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
    writer.writerow(["Time", "Production", "Consommation"])

    t = datetime(2000, 1, 1, 0, 0)
    step = timedelta(minutes=15)
    for _ in range(96):
        writer.writerow([t.strftime("%H:%M"), "", ""])
        t += step
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

    hourly_totals = totals_by_instant.resample('H').sum()
    yearly_categories = [dt.strftime('%Y-%m-%d %H:%M') for dt in hourly_totals.index]
    yearly_series = {
        'production': hourly_totals['production_kwh'].round(2).tolist(),
        'consumption': hourly_totals['consumption_kwh'].round(2).tolist(),
        'net': (hourly_totals['production_kwh'] - hourly_totals['consumption_kwh']).round(2).tolist(),
    }

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
        'yearly_categories': json.dumps(yearly_categories),
        'yearly_series': json.dumps(yearly_series),
        'avg_profile_categories': json.dumps(avg_profile_index),
        'avg_profile_series': json.dumps(avg_profile_series),
        'aggregate_metadata': aggregate_metadata,
        'pie_producers': json.dumps(pie_producers),
        'pie_consumers': json.dumps(pie_consumers),
    }

    return render(request, "core/project_analysis.html", context)
