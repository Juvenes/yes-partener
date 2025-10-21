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
from django.db import IntegrityError # Import IntegrityError
from django.contrib import messages
import json
from django.http import JsonResponse
import pandas as pd
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
    data_mode = request.POST.get('data_mode')

    if data_mode == 'timeseries_csv':
        form = MemberForm(request.POST, request.FILES)
        if form.is_valid():
            m = form.save(commit=False)
            m.project = project
            m.save()
            return redirect("project_detail", project_id=project.id)
        return HttpResponseBadRequest(f"Formulaire invalide : {form.errors.as_json()}")

    elif data_mode == 'profile_based':
        # --- Logique de génération de CSV à partir d'un profil ---
        profile_ids = request.POST.getlist('profiles')
        annual_consumption = request.POST.get('annual_consumption_kwh')
        annual_production = request.POST.get('annual_production_kwh')

        if not profile_ids:
            messages.error(request, "Veuillez sélectionner au moins un profil.")
            return redirect("project_detail", project_id=project_id)
        if not (annual_consumption or annual_production):
            messages.error(request, "Veuillez fournir une consommation ou production annuelle.")
            return redirect("project_detail", project_id=project_id)

        # Calculer les données sur 96 points
        try:
            daily_prod_kwh = (float(annual_production) if annual_production else 0.0) / 365
            daily_cons_kwh = (float(annual_consumption) if annual_consumption else 0.0) / 365
            
            profiles = Profile.objects.filter(id__in=profile_ids)
            aggregated_points = pd.DataFrame({'production': [0.0] * 96, 'consumption': [0.0] * 96})
            
            for profile in profiles:
                if profile.points and len(profile.points) == 96:
                    profile_df = pd.DataFrame(profile.points)
                    aggregated_points['production'] += profile_df['production'].astype(float)
                    aggregated_points['consumption'] += profile_df['consumption'].astype(float)

            # Normaliser le profil agrégé pour que sa somme soit 1
            if aggregated_points['production'].sum() > 0:
                aggregated_points['production'] /= aggregated_points['production'].sum()
            if aggregated_points['consumption'].sum() > 0:
                aggregated_points['consumption'] /= aggregated_points['consumption'].sum()
            
            # Appliquer les totaux journaliers
            final_production = aggregated_points['production'] * daily_prod_kwh
            final_consumption = aggregated_points['consumption'] * daily_cons_kwh

            # Créer le fichier CSV en mémoire
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Time", "Production", "Consommation"])
            t = datetime(2000, 1, 1, 0, 0)
            step = timedelta(minutes=15)
            for i in range(96):
                writer.writerow([t.strftime("%H:%M"), f"{final_production.iloc[i]:.5f}", f"{final_consumption.iloc[i]:.5f}"])
                t += step
            
            # Sauvegarder le nouveau membre et son CSV
            member = Member(
                project=project,
                name=request.POST.get('name'),
                utility=request.POST.get('utility', ''),
                data_mode='timeseries_csv',  # On normalise en mode CSV !
                annual_consumption_kwh=(float(annual_consumption) if annual_consumption else None),
                annual_production_kwh=(float(annual_production) if annual_production else None)
            )
            member.save()
            csv_file = ContentFile(output.getvalue().encode('utf-8'))
            member.timeseries_file.save(f"generated_{member.id}.csv", csv_file, save=True)

            messages.success(request, f"Le membre '{member.name}' a été créé avec un CSV généré.")

        except Exception as e:
            messages.error(request, f"Une erreur est survenue : {e}")
        
        return redirect("project_detail", project_id=project.id)

    return HttpResponseBadRequest("Mode de données non valide.")



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


def profiles_tab(request):
    """View to render the profiles tab content."""
    profiles = Profile.objects.filter(is_active=True).order_by("name")
    profile_form = ProfileForm()
    return render(request, "core/tabs/profiles_tab.html", {
        "profiles": profiles,
        "profile_form": profile_form,
    })

def global_parameters_tab(request):
    """View to render the global parameters tab content."""
    gp_form = GlobalParameterForm()
    gp_list = GlobalParameter.objects.all()
    return render(request, "core/tabs/global_parameters_tab.html", {
        "gp_form": gp_form,
        "gp_list": gp_list,
    })


def project_analysis(request, project_id):
    project = get_object_or_404(Project, pk=project_id)
    members = project.members.all()

    if not members:
        return JsonResponse({"labels": [], "datasets": []})

    # On initialise un DataFrame pandas pour agréger les données
    # 96 intervalles de 15 minutes
    analysis_df = pd.DataFrame({
        'production': [0.0] * 96,
        'consumption': [0.0] * 96,
    })

    for member in members:
        if member.data_mode == 'timeseries_csv' and member.timeseries_file:
            # On lit le fichier CSV du membre
            try:
                df = pd.read_csv(member.timeseries_file.path)
                # On s'assure que les colonnes existent
                if 'Production' in df.columns and 'Consommation' in df.columns:
                    # On somme les valeurs
                    analysis_df['production'] += df['Production'].astype(float).values
                    analysis_df['consumption'] += df['Consommation'].astype(float).values
            except Exception as e:
                # Gérer le cas où le CSV est mal formaté
                print(f"Error processing file for member {member.name}: {e}")

        elif member.data_mode == 'profile_based':
            # Logique pour les membres basés sur des profils
            total_prod = 0
            total_cons = 0

            for member_profile in member.member_profiles.all():
                profile = member_profile.profile
                if profile and profile.points and profile.is_valid_shape():
                    # On récupère les points du profil (qui sont pour 1 kWh)
                    profile_points = profile.points
                    
                    # On met à l'échelle avec la consommation/production annuelle du membre
                    if member.annual_production_kwh:
                        # Simple répartition linéaire pour l'exemple.
                        # Vous pourriez avoir une logique plus complexe.
                        prod_points = [p['production'] * member.annual_production_kwh for p in profile_points]
                        total_prod += sum(prod_points)

                    if member.annual_consumption_kwh:
                        cons_points = [p['consumption'] * member.annual_consumption_kwh for p in profile_points]
                        total_cons += sum(cons_points)
            
            # Ici, il faudrait répartir `total_prod` et `total_cons` sur les 96 points.
            # Pour l'instant, on laisse cette partie pour une V2.

    # On calcule le surplus/déficit
    analysis_df['surplus'] = analysis_df['production'] - analysis_df['consumption']

    # On prépare les données pour Chart.js
    labels = [f"{i//4:02d}:{i%4*15:02d}" for i in range(96)]
    data = {
        'labels': labels,
        'datasets': [
            {
                'label': 'Production (kWh)',
                'data': analysis_df['production'].tolist(),
                'borderColor': 'green',
                'fill': False,
            },
            {
                'label': 'Consommation (kWh)',
                'data': analysis_df['consumption'].tolist(),
                'borderColor': 'red',
                'fill': False,
            },
            {
                'label': 'Surplus (kWh)',
                'data': analysis_df['surplus'].tolist(),
                'borderColor': 'blue',
                'fill': True,
                'backgroundColor': 'rgba(0, 0, 255, 0.1)',
            }
        ]
    }
    return JsonResponse(data)


def project_analysis_page(request, project_id):
    project = get_object_or_404(Project, pk=project_id)
    # On filtre les membres qui ont bien un fichier CSV
    members = project.members.filter(data_mode='timeseries_csv').exclude(timeseries_file='')

    if not members.exists():
        messages.warning(request, "Aucun membre avec un fichier CSV n'a été trouvé pour l'analyse.")
        return redirect("project_detail", project_id=project.id)

    # Créer les timestamps pour l'axe X (96 intervalles de 15 min)
    timestamps = [
        (datetime(2000, 1, 1, 0, 0) + timedelta(minutes=15 * i)).strftime("%H:%M")
        for i in range(96)
    ]

    # --- Préparation des données pour le graphique principal ---
    series_data = []
    all_dfs = {} # Utiliser un dictionnaire pour stocker les dataframes par membre

    for member in members:
        try:
            # S'assurer que le fichier existe et n'est pas vide
            if not member.timeseries_file:
                continue

            df = pd.read_csv(member.timeseries_file.path)
            # Validation basique du CSV
            if 'Production' not in df.columns or 'Consommation' not in df.columns or len(df) != 96:
                messages.warning(request, f"Le fichier CSV pour le membre '{member.name}' est mal formaté ou incomplet (doit contenir 96 lignes et les colonnes 'Production', 'Consommation').")
                continue

            # S'assurer que les colonnes sont numériques et remplir les NaN avec 0
            df['Production'] = pd.to_numeric(df['Production'], errors='coerce').fillna(0)
            df['Consommation'] = pd.to_numeric(df['Consommation'], errors='coerce').fillna(0)

            series_data.append({'name': f"{member.name} (Prod)", 'data': df['Production'].tolist()})
            series_data.append({'name': f"{member.name} (Conso)", 'data': df['Consommation'].tolist()})
            all_dfs[member.name] = df # Ajouter le df au dictionnaire

        except Exception as e:
            messages.error(request, f"Erreur lors de la lecture du fichier pour {member.name}: {e}")
            continue

    # --- Agrégation et calcul des KPIs ---
    if not all_dfs:
         messages.error(request, "Impossible de traiter les fichiers CSV des membres. Vérifiez leur format.")
         return redirect("project_detail", project_id=project.id)

    # Concaténer tous les dataframes valides pour l'analyse globale
    global_df = pd.concat(all_dfs.values())
    total_prod_series = global_df.groupby(global_df.index)['Production'].sum()
    total_conso_series = global_df.groupby(global_df.index)['Consommation'].sum()

    # Calcul du surplus/déficit global
    surplus_series = (total_prod_series - total_conso_series).tolist()

    # --- Calcul des Chiffres Clés (KPIs) ---
    kpis = {
        'total_production': total_prod_series.sum(),
        'total_consumption': total_conso_series.sum(),
        'net_surplus': sum(surplus_series),
        'peak_production': total_prod_series.max(),
        'peak_consumption': total_conso_series.max(),
        'autonomy_coverage': (total_prod_series.sum() / total_conso_series.sum() * 100) if total_conso_series.sum() > 0 else 0,
    }

    # --- Données pour les graphiques circulaires ---
    member_totals = []
    for name, df in all_dfs.items():
        member_totals.append({
            'name': name,
            'total_prod': df['Production'].sum(),
            'total_conso': df['Consommation'].sum(),
        })

    # Trier pour obtenir les top 5
    top_producers = sorted([m for m in member_totals if m['total_prod'] > 0], key=lambda x: x['total_prod'], reverse=True)[:5]
    top_consumers = sorted([m for m in member_totals if m['total_conso'] > 0], key=lambda x: x['total_conso'], reverse=True)[:5]

    pie_producers = {'labels': [p['name'] for p in top_producers], 'data': [p['total_prod'] for p in top_producers]}
    pie_consumers = {'labels': [c['name'] for c in top_consumers], 'data': [c['total_conso'] for c in top_consumers]}

    context = {
        'project': project,
        'timestamps': json.dumps(timestamps),
        'series_data': json.dumps(series_data),
        'surplus_series': json.dumps(surplus_series),
        'kpis': kpis,
        'pie_producers': json.dumps(pie_producers),
        'pie_consumers': json.dumps(pie_consumers),
    }

    return render(request, "core/project_analysis.html", context)