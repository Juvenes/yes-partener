from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

# ---- GLOBAL SHARED PROFILE ----
class Profile(models.Model):
    name = models.CharField(max_length=200, unique=True)
    # 96 values for 24h with a 15 min interval;
    points = models.JSONField(help_text="List of 96 floats (24h, 15 min interval)")
    version = models.PositiveIntegerField(default=1)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    graph = models.ImageField(upload_to='profile_graphs/', blank=True, null=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} v{self.version}"

    def is_valid_shape(self):
        return isinstance(self.points, list) and len(self.points) == 96

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

    # Mode: profile_based
    annual_consumption_kwh = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])
    annual_production_kwh = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])

    # Stage 3 – financial data
    current_unit_price_eur_per_kwh = models.FloatField(
        "Current unit price (€/kWh)",
        default=0.0,
        blank=True,
        validators=[MinValueValidator(0.0)],
    )
    current_fixed_annual_fee_eur = models.FloatField(
        "Current fixed annual fee (€)",
        default=0.0,
        blank=True,
        validators=[MinValueValidator(0.0)],
    )
    injected_energy_kwh = models.FloatField(
        "Injected energy (kWh)",
        default=0.0,
        blank=True,
        validators=[MinValueValidator(0.0)],
    )
    injection_price_eur_per_kwh = models.FloatField(
        "Injection price (€/kWh)",
        default=0.0,
        blank=True,
        validators=[MinValueValidator(0.0)],
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