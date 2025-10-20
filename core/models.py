from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

# ---- GLOBAL SHARED PROFILE ----
class Profile(models.Model):
    KIND_CHOICES = [
        ("production", "Production"),
        ("consumption", "Consommation"),
    ]
    name = models.CharField(max_length=200, unique=True)
    kind = models.CharField(max_length=12, choices=KIND_CHOICES)
    # 96 valeurs pour 24h par pas de 15 min; doivent sommer à 1.0 pour un profil "1 kWh" (recommandé)
    points = models.JSONField(help_text="Liste de 96 floats (24h, pas 15 min)")
    version = models.PositiveIntegerField(default=1)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} [{self.kind}] v{self.version}"

    def is_valid_shape(self):
        return isinstance(self.points, list) and len(self.points) == 96

# ---- PROJECT & MEMBERS ----
class Project(models.Model):
    name = models.CharField("Nom du projet", max_length=200, unique=True)
    phase = models.PositiveSmallIntegerField(default=1, validators=[MinValueValidator(1), MaxValueValidator(4)])
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name

class Member(models.Model):
    DATA_MODE = [
        ("timeseries_csv", "Série 15-min via CSV"),
        ("profile_based", "Basé sur profil(s)"),
    ]
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="members", null=True, blank=True)
    name = models.CharField("Nom", max_length=200)
    utility = models.CharField("Utilité", max_length=200, blank=True)
    data_mode = models.CharField(max_length=20, choices=DATA_MODE, default="timeseries_csv")

    # --- Mode CSV ---
    # CSV attendu: colonnes: Time,Production,Consommation ; Time = HH:MM à pas 15 min
    timeseries_file = models.FileField(upload_to="timeseries/", blank=True, null=True)

    # --- Mode Profil ---
    # Totaux pour le scaling (tu pourras choisir d'en utiliser un seul)
    annual_consumption_kwh = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])
    daily_consumption_kwh = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])
    annual_production_kwh = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])
    daily_production_kwh = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])

    class Meta:
        unique_together = ("project", "name")
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.project.name})"

class MemberProfile(models.Model):
    """Lien N:N entre Member et Profile (un membre peut avoir plusieurs profils).
    scale_factor est optionnel (par ex. pour pondérer plusieurs profils au sein d'un même membre).
    """
    member = models.ForeignKey(Member, on_delete=models.CASCADE, related_name="member_profiles")
    profile = models.ForeignKey(Profile, on_delete=models.PROTECT, related_name="member_profiles")
    scale_factor = models.FloatField(default=1.0, validators=[MinValueValidator(0.0)])

    class Meta:
        unique_together = ("member", "profile")

class GlobalParameter(models.Model):
    """Petit KV store global pour paramètres partagés (JSON ou texte)."""
    key = models.CharField(max_length=100, unique=True)
    value = models.JSONField(blank=True, null=True)  # tu peux aussi y mettre un dict
    note = models.CharField(max_length=255, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["key"]

    def __str__(self):
        return self.key