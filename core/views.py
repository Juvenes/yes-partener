from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_http_methods
from django.http import HttpResponseBadRequest, HttpResponse
from dataclasses import asdict
from typing import Dict, List
from django.db.models import Q
from .models import (
    Project,
    Member,
    Dataset,
    GlobalParameter,
    StageTwoScenario,
    StageThreeScenario,
)
from .forms import (
    ProjectForm,
    MemberForm,
    DatasetForm,
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
    build_indexed_template,
    build_metadata,
    parse_member_timeseries,
    _normalise_to_reference_year,
)
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

    daily_totals = combined_df.copy()
    daily_totals['day'] = daily_totals['timestamp'].dt.date
    daily_totals = daily_totals.groupby('day')[['production_kwh', 'consumption_kwh']].sum().sort_index()
    recent_daily = daily_totals.tail(30)
    daily_categories = [day.strftime('%Y-%m-%d') for day in recent_daily.index]
    daily_series = {
        'production': [round(val, 2) for val in recent_daily['production_kwh']],
        'consumption': [round(val, 2) for val in recent_daily['consumption_kwh']],
    }

    monthly_categories = [f"{int(month):02d}" for month in totals_by_month.index]
    monthly_series = {
        'production': [round(val, 1) for val in totals_by_month['production_kwh']],
        'consumption': [round(val, 1) for val in totals_by_month['consumption_kwh']],
        'net': [round(p - c, 1) for p, c in zip(totals_by_month['production_kwh'], totals_by_month['consumption_kwh'])],
    }

    monthly_detail = []
    for month, month_df in combined_df.groupby(combined_df['timestamp'].dt.month):
        by_day = month_df.groupby(month_df['timestamp'].dt.day)[['production_kwh', 'consumption_kwh']].sum().sort_index()
        monthly_detail.append({
            'month': f"{int(month):02d}",
            'categories': [int(day) for day in by_day.index],
            'series': {
                'production': [round(val, 2) for val in by_day['production_kwh']],
                'consumption': [round(val, 2) for val in by_day['consumption_kwh']],
            },
        })
    monthly_detail.sort(key=lambda item: item['month'])

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
            "daily_categories": daily_categories,
            "daily_series": daily_series,
            "monthly_categories": monthly_categories,
            "monthly_series": monthly_series,
            "monthly_detail": monthly_detail,
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


