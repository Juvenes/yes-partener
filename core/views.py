from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_http_methods
from django.http import HttpResponseBadRequest, HttpResponse
from dataclasses import asdict
from typing import Dict, List
from django.db.models import Q
from .models import Project, Member, Dataset, GlobalParameter, StageTwoScenario
from .forms import (
    ProjectForm,
    MemberForm,
    DatasetForm,
    GlobalParameterForm,
    StageTwoScenarioForm,
    StageThreeTariffForm,
    CommunityOptimizationForm,
)
from .stage2 import (
    evaluate_sharing as evaluate_stage_two,
    build_iteration_configs,
    load_project_timeseries as load_stage_two_timeseries,
)
from .stage3 import (
    optimise_internal_price,
    build_member_tariffs,
    compute_baselines,
)
from .timeseries import (
    TimeseriesError,
    attach_calendar_index,
    build_indexed_template,
    build_metadata,
    parse_member_timeseries,
    _normalise_to_reference_year,
)
import calendar
import csv
from datetime import datetime, timedelta
import json
from django.core.files.base import ContentFile
import io
from django.contrib import messages
from django.utils.text import slugify
import pandas as pd


def _parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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

# Central dataset management page
def dataset_list(request):
    query = request.GET.get("q", "").strip()
    datasets = Dataset.objects.all().order_by("name")
    if query:
        datasets = datasets.filter(
            Q(name__icontains=query) | Q(tags__icontains=query)
        )
    dataset_form = DatasetForm()
    return render(
        request,
        "core/datasets.html",
        {
            "datasets": datasets,
            "dataset_form": dataset_form,
            "query": query,
        },
    )


@require_http_methods(["POST"])
def dataset_create(request):
    form = DatasetForm(request.POST, request.FILES)
    if not form.is_valid():
        messages.error(request, "Formulaire dataset invalide : vérifiez les champs et le fichier.")
        return redirect("datasets")

    upload = form.cleaned_data.get("source_file")
    if not upload:
        messages.error(request, "Veuillez sélectionner un fichier de données.")
        return redirect("datasets")

    try:
        parse_result = parse_member_timeseries(upload)
    except TimeseriesError as exc:
        messages.error(request, f"Import impossible : {exc}")
        return redirect("datasets")

    df = parse_result.data.copy()
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    ordered = [
        "timestamp",
        "month",
        "week_of_month",
        "weekday",
        "quarter_index",
        "production_kwh",
        "consumption_kwh",
    ]
    for col in ordered:
        if col not in df.columns:
            df[col] = None
    df = df[ordered]

    normalized_buffer = io.BytesIO()
    df.to_excel(normalized_buffer, index=False)
    normalized_buffer.seek(0)

    tags_raw = form.cleaned_data.get("tags") or ""
    tag_list = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]

    dataset = Dataset(
        name=form.cleaned_data["name"],
        tags=tag_list,
        metadata=asdict(parse_result.metadata),
    )

    upload.seek(0)
    dataset.source_file.save(upload.name or f"dataset_{dataset.name}.csv", upload, save=False)
    dataset.normalized_file.save(
        f"{slugify(dataset.name)}_normalized.xlsx",
        ContentFile(normalized_buffer.getvalue()),
        save=False,
    )
    dataset.save()

    totals = parse_result.metadata.totals or {}
    messages.success(
        request,
        (
            f"Dataset '{dataset.name}' importé : {parse_result.metadata.row_count} lignes, "
            f"Conso {totals.get('consumption_kwh', 0):.1f} kWh, Prod {totals.get('production_kwh', 0):.1f} kWh."
        ),
    )
    for warning in parse_result.metadata.warnings:
        messages.warning(request, warning)

    return redirect("datasets")

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
        label = request.POST.get("label", "").strip()
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
        buffer.seek(0)

        response = HttpResponse(buffer.getvalue(), content_type="text/csv")
        filename = slugify(label or upload.name.rsplit(".", 1)[0] or "timeseries")
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
        {"preview_rows": preview},
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
    members = project.members.all().select_related("dataset")
    member_form = MemberForm()
    member_form.fields["dataset"].queryset = Dataset.objects.all()
    datasets = Dataset.objects.all()
    return render(
        request,
        "core/project_detail.html",
        {
            "project": project,
            "members": members,
            "member_form": member_form,
            "datasets": datasets,
        },
    )

@require_http_methods(["POST"])
def member_create(request, project_id):
    project = get_object_or_404(Project, pk=project_id)
    form = MemberForm(request.POST)
    form.fields["dataset"].queryset = Dataset.objects.all()
    if not form.is_valid():
        messages.error(request, f"Formulaire invalide : {form.errors.as_json()}")
        return redirect("project_detail", project_id=project.id)

    member = form.save(commit=False)
    member.project = project

    metadata = {} if member.dataset.metadata is None else member.dataset.metadata
    totals = metadata.get("totals") or {}

    if member.annual_consumption_kwh is None:
        member.annual_consumption_kwh = _parse_float(totals.get("consumption_kwh"))
    if member.annual_production_kwh is None:
        member.annual_production_kwh = _parse_float(totals.get("production_kwh"))

    member.save()

    row_count = metadata.get("row_count") if isinstance(metadata, dict) else None
    messages.success(
        request,
        (
            f"Le membre '{member.name}' a été ajouté avec le dataset '{member.dataset.name}'. "
            f"{row_count or '—'} lignes analysées."
        ),
    )

    for warning in metadata.get("warnings", []) if isinstance(metadata, dict) else []:
        messages.warning(request, warning)

    return redirect("project_detail", project_id=project.id)


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
    writer.writerow(["Date+Quart time", "consumption", "injection"])

    base = datetime(2024, 1, 1, 0, 0)
    step = timedelta(minutes=15)
    for _ in range(8):
        writer.writerow([base.strftime("%Y-%m-%d %H:%M"), "", ""])
        base += step
    writer.writerow(["...", "", ""])
    writer.writerow(["(35 040 lignes sur une année complète)", "", ""])
    return response

def project_analysis_page(request, project_id):
    project = get_object_or_404(Project, pk=project_id)
    members = project.members.select_related("dataset")

    if not members.exists():
        messages.warning(request, "Aucun membre avec un dataset n'a été trouvé pour l'analyse.")
        return redirect("project_detail", project_id=project.id)

    combined_frames = []
    member_stats = []

    for member in members:
        dataset = getattr(member, "dataset", None)
        if not dataset or not dataset.normalized_file:
            messages.warning(request, f"{member.name} : dataset incomplet ou sans fichier normalisé.")
            continue
        try:
            parse_result = parse_member_timeseries(dataset.normalized_file.path)
        except (TimeseriesError, FileNotFoundError) as exc:
            messages.error(request, f"{member.name} : {exc}")
            continue

        df = parse_result.data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df['member'] = member.name
        combined_frames.append(df)

        metadata_dict = dataset.metadata or asdict(parse_result.metadata)
        total_production = float(df['production_kwh'].sum())
        total_consumption = float(df['consumption_kwh'].sum())

        totals_section = metadata_dict.setdefault('totals', {})
        totals_section['production_kwh'] = total_production
        totals_section['consumption_kwh'] = total_consumption

        if not metadata_dict.get('start') and not df['timestamp'].empty:
            metadata_dict['start'] = df['timestamp'].min().isoformat()
        if not metadata_dict.get('end') and not df['timestamp'].empty:
            metadata_dict['end'] = df['timestamp'].max().isoformat()

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

    normalized_ts, normalized_year = _normalise_to_reference_year(combined_df['timestamp'])
    combined_df['timestamp'] = normalized_ts
    combined_df = attach_calendar_index(combined_df)
    combined_df = combined_df.sort_values(['timestamp', 'member'])

    totals_by_instant = combined_df.groupby('timestamp')[['production_kwh', 'consumption_kwh']].sum()
    totals_by_instant.index = pd.to_datetime(totals_by_instant.index)
    totals_by_instant = totals_by_instant.sort_index()

    totals_by_month = combined_df.groupby(combined_df['timestamp'].dt.month)[['production_kwh', 'consumption_kwh']].sum().sort_index()

    member_totals = combined_df.groupby('member')[['production_kwh', 'consumption_kwh']].sum()
    member_totals = member_totals.reset_index().sort_values('member')

    total_production_all = float(totals_by_instant['production_kwh'].sum())
    total_consumption_all = float(totals_by_instant['consumption_kwh'].sum())

    avg_profile = combined_df.copy()
    avg_profile['hour'] = avg_profile['timestamp'].dt.hour + avg_profile['timestamp'].dt.minute / 60
    avg_profile = avg_profile.groupby('hour')[['production_kwh', 'consumption_kwh']].mean().reset_index()

    avg_profile_categories = [f"{hour:.2f}" for hour in avg_profile['hour']]
    avg_profile_series = {
        'production': [round(val, 3) for val in avg_profile['production_kwh']],
        'consumption': [round(val, 3) for val in avg_profile['consumption_kwh']],
    }

    monthly_categories = [f"{int(month):02d}" for month in totals_by_month.index]
    monthly_series = {
        'production': [round(val, 1) for val in totals_by_month['production_kwh']],
        'consumption': [round(val, 1) for val in totals_by_month['consumption_kwh']],
        'net': [round(p - c, 1) for p, c in zip(totals_by_month['production_kwh'], totals_by_month['consumption_kwh'])],
    }

    monthly_detail = []
    for month, month_df in combined_df.groupby(combined_df['timestamp'].dt.month):
        days = sorted(month_df['timestamp'].dt.day.unique())
        by_day = month_df.groupby(month_df['timestamp'].dt.day)[['production_kwh', 'consumption_kwh']].sum().reindex(days, fill_value=0)

        production_series = []
        consumption_series = []

        for member_name, member_df in month_df.groupby('member'):
            member_day = (
                member_df.groupby(member_df['timestamp'].dt.day)[['production_kwh', 'consumption_kwh']]
                .sum()
                .reindex(days, fill_value=0)
            )
            production_series.append({
                'name': member_name,
                'data': [round(val, 2) for val in member_day['production_kwh']],
            })
            consumption_series.append({
                'name': member_name,
                'data': [round(val, 2) for val in member_day['consumption_kwh']],
            })

        totals_production = [round(val, 2) for val in by_day['production_kwh']]
        totals_consumption = [round(val, 2) for val in by_day['consumption_kwh']]

        monthly_detail.append({
            'month': f"{int(month):02d}",
            'label': calendar.month_name[int(month)] or f"Mois {int(month):02d}",
            'categories': [int(day) for day in by_day.index],
            'production_series': production_series,
            'consumption_series': consumption_series,
            'totals': {
                'production': totals_production,
                'consumption': totals_consumption,
                'net': [round(p - c, 2) for p, c in zip(totals_production, totals_consumption)],
            },
        })
    monthly_detail.sort(key=lambda item: item['month'])

    combined_df['week'] = combined_df['timestamp'].dt.isocalendar().week.astype(int)
    weeks = sorted(combined_df['week'].unique())
    weekly_totals_df = (
        combined_df.groupby('week')[['production_kwh', 'consumption_kwh']]
        .sum()
        .reindex(weeks, fill_value=0)
    )
    weekly_totals = {
        'production': [round(val, 2) for val in weekly_totals_df['production_kwh']],
        'consumption': [round(val, 2) for val in weekly_totals_df['consumption_kwh']],
        'net': [round(p - c, 2) for p, c in zip(weekly_totals_df['production_kwh'], weekly_totals_df['consumption_kwh'])],
    }

    weekly_member_production = []
    weekly_member_consumption = []
    for member_name, member_df in combined_df.groupby('member'):
        member_weekly = (
            member_df.groupby('week')[['production_kwh', 'consumption_kwh']]
            .sum()
            .reindex(weeks, fill_value=0)
        )
        weekly_member_production.append({
            'name': member_name,
            'data': [round(val, 2) for val in member_weekly['production_kwh']],
        })
        weekly_member_consumption.append({
            'name': member_name,
            'data': [round(val, 2) for val in member_weekly['consumption_kwh']],
        })

    pie_producers = [
        {'name': row['member'], 'y': round(row['production_kwh'], 2)}
        for _, row in member_totals.sort_values('production_kwh', ascending=False).head(5).iterrows()
    ]
    pie_consumers = [
        {'name': row['member'], 'y': round(row['consumption_kwh'], 2)}
        for _, row in member_totals.sort_values('consumption_kwh', ascending=False).head(5).iterrows()
    ]

    aggregate_metadata = build_metadata(
        combined_df,
        {'timestamp': 'timestamp', 'production': 'production_kwh', 'consumption': 'consumption_kwh'},
        file_type='project_analysis',
        totals_override={'production_kwh': total_production_all, 'consumption_kwh': total_consumption_all},
        normalized_year=normalized_year,
    )

    kpis = {
        'total_production': total_production_all,
        'total_consumption': total_consumption_all,
        'net_surplus': total_production_all - total_consumption_all,
        'autonomy_coverage': (total_production_all / total_consumption_all * 100) if total_consumption_all > 0 else 0.0,
    }

    member_stats_sorted = sorted(
        member_stats,
        key=lambda item: item['metadata'].get('totals', {}).get('consumption_kwh', 0),
        reverse=True,
    )

    return render(
        request,
        "core/project_analysis.html",
        {
            "project": project,
            "aggregate_metadata": asdict(aggregate_metadata),
            "member_stats": member_stats_sorted,
            "totals_by_instant": totals_by_instant.to_dict(orient="records"),
            "totals_by_month": totals_by_month.to_dict(orient="index"),
            "member_totals": member_totals.to_dict(orient="records"),
            "total_production_all": total_production_all,
            "total_consumption_all": total_consumption_all,
            "avg_profile_categories": avg_profile_categories,
            "avg_profile_series": avg_profile_series,
            "monthly_categories": monthly_categories,
            "monthly_series": monthly_series,
            "monthly_detail": monthly_detail,
            "weekly_categories": [int(week) for week in weeks],
            "weekly_totals": weekly_totals,
            "weekly_member_production": weekly_member_production,
            "weekly_member_consumption": weekly_member_consumption,
            "pie_producers": pie_producers,
            "pie_consumers": pie_consumers,
            "kpis": kpis,
        },
    )


def project_stage2(request, project_id):
    project = get_object_or_404(Project, pk=project_id)
    all_members = list(project.members.all().order_by("name"))
    members_with_series = [
        member for member in all_members if getattr(member, "dataset", None) and member.dataset.normalized_file
    ]
    missing_members = [member for member in all_members if member not in members_with_series]

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

    def build_stage2_visuals(evaluation):
        timeline = evaluation.timeline.copy()
        if timeline.empty:
            return {"has_data": False}

        if hasattr(timeline["timestamp"], "dt"):
            timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], errors="coerce")
        timeline = timeline.dropna(subset=["timestamp"])
        timeline = timeline.sort_values("timestamp")
        timeline["month_num"] = timeline["timestamp"].dt.month
        timeline["month_label"] = timeline["month_num"].apply(lambda m: calendar.month_abbr[int(m)])
        timeline["week_num"] = timeline["timestamp"].dt.isocalendar().week.astype(int)

        monthly = (
            timeline.groupby(["month_num", "month_label"])[
                [
                    "production_total_kwh",
                    "community_allocated_kwh",
                    "remaining_production_kwh",
                    "remaining_consumption_kwh",
                ]
            ]
            .sum()
            .reset_index()
            .sort_values("month_num")
        )
        months_order = list(monthly["month_num"].astype(int))
        month_labels = list(monthly["month_label"])

        member_monthly = []
        member_weekly = []

        for member in members_with_series:
            key = f"member_{member.id}_community_kwh"
            if key not in timeline.columns:
                continue
            member_month = timeline.groupby("month_num")[key].sum().reindex(months_order, fill_value=0.0)
            member_monthly.append(
                {
                    "member": member.name,
                    "values": [float(value) for value in member_month.tolist()],
                }
            )

            weekly_group = timeline.groupby("week_num")[key].sum()
            weeks_sorted = sorted(weekly_group.index.tolist())
            member_weekly.append(
                {
                    "member": member.name,
                    "weeks": weeks_sorted,
                    "values": [float(weekly_group.get(week, 0.0)) for week in weeks_sorted],
                }
            )

        weekly_totals = (
            timeline.groupby("week_num")[["community_allocated_kwh", "production_total_kwh", "remaining_consumption_kwh"]]
            .sum()
            .reset_index()
            .sort_values("week_num")
        )

        samples = timeline[timeline["community_allocated_kwh"] > 0].head(6)
        if samples.empty:
            samples = timeline.head(3)

        sample_rows = []
        for row in samples.itertuples():
            record = {
                "timestamp": getattr(row, "timestamp").isoformat() if hasattr(getattr(row, "timestamp"), "isoformat") else str(getattr(row, "timestamp")),
                "production": float(getattr(row, "production_total_kwh", 0.0)),
                "consumption": float(getattr(row, "consumption_total_kwh", 0.0)),
                "allocated": float(getattr(row, "community_allocated_kwh", 0.0)),
                "remaining_production": float(getattr(row, "remaining_production_kwh", 0.0)),
                "remaining_consumption": float(getattr(row, "remaining_consumption_kwh", 0.0)),
                "members": [],
            }
            for member in members_with_series:
                community_key = f"member_{member.id}_community_kwh"
                external_key = f"member_{member.id}_external_kwh"
                prod_unused_key = f"member_{member.id}_production_unused_kwh"
                record["members"].append(
                    {
                        "name": member.name,
                        "community": float(getattr(row, community_key, 0.0)),
                        "external": float(getattr(row, external_key, 0.0)),
                        "unused_prod": float(getattr(row, prod_unused_key, 0.0)),
                    }
                )
            sample_rows.append(record)

        iteration_chart = [
            {
                "label": f"Itération {stat.order} — {key_labels.get(stat.key_type, stat.key_type)}",
                "value": float(stat.allocated_kwh),
            }
            for stat in evaluation.iteration_stats
        ]

        return {
            "has_data": True,
            "months": month_labels,
            "monthly_totals": [
                {
                    "label": label,
                    "production": float(monthly.iloc[idx]["production_total_kwh"]),
                    "community": float(monthly.iloc[idx]["community_allocated_kwh"]),
                    "remaining_production": float(monthly.iloc[idx]["remaining_production_kwh"]),
                    "remaining_consumption": float(monthly.iloc[idx]["remaining_consumption_kwh"]),
                }
                for idx, label in enumerate(month_labels)
            ],
            "member_monthly": member_monthly,
            "weekly_totals": [
                {
                    "week": int(row.week_num),
                    "community": float(row.community_allocated_kwh),
                    "production": float(row.production_total_kwh),
                    "remaining_consumption": float(row.remaining_consumption_kwh),
                }
                for row in weekly_totals.itertuples()
            ],
            "member_weekly": member_weekly,
            "iteration_chart": iteration_chart,
            "samples": sample_rows,
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
                    "viz_payload": json.dumps(build_stage2_visuals(evaluation)),
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
    members_with_series = [
        member for member in members if getattr(member, "dataset", None) and member.dataset.normalized_file
    ]

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
    member_forms = {
        member.id: StageThreeTariffForm(instance=member, prefix=f"member-{member.id}")
        for member in members
    }

    optimisation_form = CommunityOptimizationForm(prefix="opt")
    optimisation_result = None
    baselines = {}
    load_warnings: List[str] = []
    timeseries_df = None

    try:
        timeseries_df, load_warnings = load_stage_two_timeseries(members)
    except ValueError as exc:
        messages.error(request, str(exc))
    except Exception as exc:  # pragma: no cover - safety net for UI feedback
        messages.error(request, f"Impossible de charger les séries temporelles : {exc}")

    tariffs = build_member_tariffs(members)
    baselines = compute_baselines(timeseries_df, tariffs)

    if request.method == "POST" and "optimize" in request.POST:
        optimisation_form = CommunityOptimizationForm(request.POST, prefix="opt")
        if optimisation_form.is_valid():
            if timeseries_df is None or timeseries_df.empty:
                messages.error(
                    request,
                    "Aucune donnée temporelle normalisée n'est disponible pour optimiser le prix communautaire.",
                )
            else:
                data = optimisation_form.cleaned_data
                optimisation_result = optimise_internal_price(
                    tariffs=tariffs,
                    timeseries=timeseries_df,
                    community_fee_eur_per_kwh=data.get("community_fee_eur_per_kwh") or 0.0,
                    community_type=data.get("community_type") or "public_grid",
                    reduced_distribution=data.get("reduced_distribution_eur_per_kwh"),
                    reduced_transport=data.get("reduced_transport_eur_per_kwh"),
                )
        else:
            messages.error(request, "Corrigez les paramètres de la communauté pour lancer l'optimisation.")

    outcome_lookup: Dict[int, object] = {}
    if optimisation_result and optimisation_result.member_outcomes:
        outcome_lookup = {out.member_id: out for out in optimisation_result.member_outcomes}

    total_baseline_cost = sum(result.cost_eur for result in baselines.values()) if baselines else 0.0
    total_consumption = sum(result.consumption_kwh for result in baselines.values()) if baselines else 0.0
    avg_baseline_cost = total_baseline_cost / total_consumption if total_consumption > 0 else 0.0

    member_cards = []
    for tariff in tariffs:
        baseline = baselines.get(tariff.member_id)
        member_cards.append(
            {
                "tariff": tariff,
                "baseline": baseline,
                "form": member_forms.get(tariff.member_id),
                "community": outcome_lookup.get(tariff.member_id),
            }
        )

    return render(
        request,
        "core/project_stage3.html",
        {
            "project": project,
            "member_cards": member_cards,
            "optimisation_form": optimisation_form,
            "optimisation_result": optimisation_result,
            "community_outcomes": outcome_lookup,
            "total_baseline_cost": total_baseline_cost,
            "total_consumption": total_consumption,
            "avg_baseline_cost": avg_baseline_cost,
            "load_warnings": load_warnings,
        },
    )


@require_http_methods(["POST"])
def stage3_member_update(request, project_id, member_id):
    project = get_object_or_404(Project, pk=project_id)
    member = get_object_or_404(Member, pk=member_id, project=project)
    form = StageThreeTariffForm(
        request.POST, instance=member, prefix=f"member-{member.id}"
    )
    if form.is_valid():
        form.save()
        messages.success(request, f"Structure tarifaire mise à jour pour {member.name}.")
    else:
        messages.error(request, f"Impossible de mettre à jour {member.name} : {form.errors.as_json()}")
    return redirect("project_stage3", project_id=project.id)


