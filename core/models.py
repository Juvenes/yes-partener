from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

# ---- GLOBAL SHARED PROFILE ----
class Profile(models.Model):
    PROFILE_TYPES = [
        ("consumption", "Consommation"),
        ("production", "Production"),
    ]

    name = models.CharField(max_length=200, unique=True)
    profile_type = models.CharField(max_length=20, choices=PROFILE_TYPES, default="consumption")
    # Complete year of quarter-hour values (~35k rows) stored as JSON.
    points = models.JSONField(help_text="List of quarter-hour values for a full year")
    metadata = models.JSONField(blank=True, null=True)
    version = models.PositiveIntegerField(default=1)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    graph = models.ImageField(upload_to='profile_graphs/', blank=True, null=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} v{self.version}"

    def is_valid_shape(self):
        return isinstance(self.points, list) and len(self.points) >= 96

    @property
    def point_count(self):
        return len(self.points) if isinstance(self.points, list) else 0

# ---- PROJECT & MEMBERS ----
class Project(models.Model):
    name = models.CharField("Project name", max_length=200, unique=True)
    phase = models.PositiveSmallIntegerField(default=1, validators=[MinValueValidator(1), MaxValueValidator(4)])
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name

class Member(models.Model):
    DATA_MODE_CHOICES = [
        ("timeseries_csv", "Série 15-min via CSV"),
        ("profile_based", "Basé sur profil(s)"),
    ]

    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="members", null=True, blank=True)
    name = models.CharField("Name", max_length=200)
    utility = models.CharField("Utility", max_length=200, blank=True)
    
    data_mode = models.CharField(max_length=20, choices=DATA_MODE_CHOICES, default="timeseries_csv")

    # Mode: timeseries_csv
    timeseries_file = models.FileField(upload_to="timeseries/", blank=True, null=True)
    timeseries_metadata = models.JSONField(blank=True, null=True)

    # Mode: profile_based
    annual_consumption_kwh = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])
    annual_production_kwh = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])

    # Stage 3 cost inputs
    current_unit_price_eur_per_kwh = models.FloatField(
        default=0.25,
        validators=[MinValueValidator(0.0)],
        help_text="Prix unitaire actuel payé au fournisseur (€/kWh).",
    )
    current_fixed_fee_eur = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0)],
        help_text="Partie fixe annuelle de la facture actuelle (€/an).",
    )
    injection_annual_kwh = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0)],
        help_text="Energie annuelle injectée sur le réseau (kWh).",
    )
    injection_unit_price_eur_per_kwh = models.FloatField(
        default=0.05,
        validators=[MinValueValidator(0.0)],
        help_text="Tarif de rachat pour l'injection (€/kWh).",
    )
    
    class Meta:
        unique_together = ("project", "name")
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.project.name})"

class MemberProfile(models.Model):
    """Lien N:N entre Membre et Profil (un membre peut avoir plusieurs profils)."""
    member = models.ForeignKey(Member, on_delete=models.CASCADE, related_name="member_profiles")
    profile = models.ForeignKey(Profile, on_delete=models.PROTECT, related_name="member_profiles")

    class Meta:
        unique_together = ("member", "profile")

class GlobalParameter(models.Model):
    """Small global KV store for shared parameters (JSON or text)."""
    key = models.CharField(max_length=100, unique=True)
    value = models.JSONField(blank=True, null=True)  # You can also put a dict in it
    note = models.CharField(max_length=255, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["key"]

    def __str__(self):
        return self.key


class StageThreeScenario(models.Model):
    FEE_ALLOCATION_CHOICES = [
        ("all_members", "Répartition égale (tous les membres)"),
        ("participants", "Répartition égale (participants)"),
        ("consumption", "Proportionnelle à l'énergie communautaire"),
    ]

    TARIFF_CONTEXT_CHOICES = [
        ("community_grid", "Communauté via réseau public"),
        ("community_same_site", "Communauté même bâtiment"),
        ("traditional", "Facture fournisseur de référence"),
    ]

    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="stage3_scenarios",
    )
    name = models.CharField(max_length=200)
    community_price_eur_per_kwh = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0)],
        help_text="Prix cible de l'énergie communautaire (€/kWh).",
    )
    price_min_eur_per_kwh = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0)],
        help_text="Borne minimum pour optimiser le prix communautaire (€/kWh).",
    )
    price_max_eur_per_kwh = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0)],
        help_text="Borne maximum pour optimiser le prix communautaire (€/kWh).",
    )
    price_step_eur_per_kwh = models.FloatField(
        default=0.005,
        validators=[MinValueValidator(0.0001)],
        help_text="Pas utilisé lors de l'exploration du prix communautaire.",
    )
    default_share = models.FloatField(
        default=1.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Part de consommation couverte par défaut par l'énergie communautaire (0-1).",
    )
    coverage_cap = models.FloatField(
        default=1.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Part maximale pouvant être couverte (0-1).",
    )
    community_fixed_fee_total_eur = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0)],
        help_text="Coût fixe annuel de la communauté à répartir (€/an).",
    )
    community_per_member_fee_eur = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0)],
        help_text="Frais annuels individuels (€/an) pour les membres participants.",
    )
    community_variable_fee_eur_per_kwh = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0)],
        help_text="Frais variables appliqués à chaque kWh communautaire (€/kWh).",
    )
    community_injection_price_eur_per_kwh = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0)],
        help_text="Tarif interne pour rémunérer l'injection (€/kWh). Laisser vide pour utiliser le tarif membre.",
    )
    fee_allocation = models.CharField(
        max_length=20,
        choices=FEE_ALLOCATION_CHOICES,
        default="participants",
    )
    tariff_context = models.CharField(
        max_length=40,
        choices=TARIFF_CONTEXT_CHOICES,
        default="community_grid",
        help_text="Cadre réglementaire utilisé pour expliquer les composantes de coût.",
    )
    notes = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("project", "name")
        ordering = ["project", "name"]

    def __str__(self):
        return f"{self.project.name} – {self.name}"


class StageThreeScenarioMember(models.Model):
    scenario = models.ForeignKey(
        StageThreeScenario,
        on_delete=models.CASCADE,
        related_name="member_settings",
    )
    member = models.ForeignKey(
        Member,
        on_delete=models.CASCADE,
        related_name="stage3_settings",
    )
    share_override = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Valeur utilisée par défaut pour ce membre (0-1).",
    )
    min_share = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Bornes minimales pour les optimisations (0-1).",
    )
    max_share = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Bornes maximales pour les optimisations (0-1).",
    )

    class Meta:
        unique_together = ("scenario", "member")
        ordering = ["scenario", "member__name"]

    def __str__(self):
        return f"{self.scenario.name} → {self.member.name}"