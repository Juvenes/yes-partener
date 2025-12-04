from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator


def _random_color():
    """Generate a pleasant pastel-like color for tag pills."""

    import random

    base = random.randint(80, 200)
    r = base
    g = min(220, base + random.randint(-20, 35))
    b = min(230, base + random.randint(10, 50))
    return f"#{r:02x}{g:02x}{b:02x}"


class Tag(models.Model):
    name = models.CharField(max_length=100, unique=True)
    color = models.CharField(max_length=7, default=_random_color)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name

# ---- GLOBAL SHARED PROFILE ----
class Dataset(models.Model):
    name = models.CharField(max_length=200, unique=True)
    tags = models.ManyToManyField(Tag, blank=True, related_name="datasets")
    source_file = models.FileField(upload_to="datasets/source/")
    normalized_file = models.FileField(upload_to="datasets/normalized/")
    metadata = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name

    def tag_list(self):
        return list(self.tags.all())

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
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="members", null=True, blank=True)
    dataset = models.ForeignKey(Dataset, on_delete=models.PROTECT, related_name="members", null=True, blank=True)
    name = models.CharField("Name", max_length=200)
    utility = models.CharField("Utility", max_length=200, blank=True)
    tags = models.ManyToManyField(Tag, blank=True, related_name="members")

    annual_consumption_kwh = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])
    annual_production_kwh = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])

    # Stage 3 tariff components (€/kWh)
    supplier_energy_price_eur_per_kwh = models.FloatField(
        default=0.1308,
        validators=[MinValueValidator(0.0)],
        help_text="Prix énergie facturé par le fournisseur (€/kWh).",
    )
    distribution_tariff_eur_per_kwh = models.FloatField(
        default=0.1018,
        validators=[MinValueValidator(0.0)],
        help_text="Tarif de distribution (DSO) (€/kWh).",
    )
    transport_tariff_eur_per_kwh = models.FloatField(
        default=0.0281,
        validators=[MinValueValidator(0.0)],
        help_text="Tarif de transport (TSO) (€/kWh).",
    )
    green_support_eur_per_kwh = models.FloatField(
        default=0.0284,
        validators=[MinValueValidator(0.0)],
        help_text="Soutien énergie verte (€/kWh).",
    )
    access_fee_eur_per_kwh = models.FloatField(
        default=0.0008,
        validators=[MinValueValidator(0.0)],
        help_text="Redevance d'accès (€/kWh).",
    )
    special_excise_eur_per_kwh = models.FloatField(
        default=0.0142,
        validators=[MinValueValidator(0.0)],
        help_text="Accise spéciale (€/kWh).",
    )
    energy_contribution_eur_per_kwh = models.FloatField(
        default=0.0019,
        validators=[MinValueValidator(0.0)],
        help_text="Contribution énergie (€/kWh).",
    )
    injection_price_eur_per_kwh = models.FloatField(
        default=0.05,
        validators=[MinValueValidator(0.0)],
        help_text="Tarif d'injection (€/kWh).",
    )
    
    class Meta:
        unique_together = ("project", "name")
        ordering = ["name"]

    def __str__(self):
        if self.project:
            return f"{self.name} ({self.project.name})"
        return self.name

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


class StageTwoScenario(models.Model):
    KEY_TYPE_CHOICES = [
        ("equal", "Clé part égale"),
        ("percentage", "Clé pourcentage fixe"),
        ("proportional", "Clé proportionnelle à la consommation"),
    ]

    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="stage2_scenarios",
    )
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    iterations = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("project", "name")
        ordering = ["project", "name"]

    def __str__(self):
        return f"{self.project.name} – {self.name}"

    def iteration_configs(self):
        configs = []
        allowed = {choice[0] for choice in self.KEY_TYPE_CHOICES}
        raw_iterations = self.iterations or []
        for index, payload in enumerate(raw_iterations, start=1):
            if not isinstance(payload, dict):
                continue
            key_type = payload.get("key_type")
            if key_type not in allowed:
                continue
            order = payload.get("order") or index
            try:
                order = int(order)
            except (TypeError, ValueError):
                order = index

            raw_percentages = payload.get("percentages") or {}
            percentages = {}
            if isinstance(raw_percentages, dict):
                for member_id, value in raw_percentages.items():
                    try:
                        member_key = int(member_id)
                        percentages[member_key] = float(value)
                    except (TypeError, ValueError):
                        continue

            configs.append(
                {
                    "order": order,
                    "key_type": key_type,
                    "percentages": percentages,
                }
            )

        configs.sort(key=lambda item: item.get("order", 0))
        return configs


class StageThreeScenario(models.Model):
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
