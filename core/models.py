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
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="members", null=True, blank=True)
    name = models.CharField("Name", max_length=200)
    utility = models.CharField("Utility", max_length=200, blank=True)
    # Expected CSV: Time,Production,Consumption; Time = HH:MM with a 15 min interval
    timeseries_file = models.FileField(upload_to="timeseries/", blank=True, null=True)

    class Meta:
        unique_together = ("project", "name")
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.project.name})"

class MemberProfile(models.Model):
    """N:N link between Member and Profile (a member can have multiple profiles).
    scale_factor is optional (e.g. to weight multiple profiles within the same member).
    """
    member = models.ForeignKey(Member, on_delete=models.CASCADE, related_name="member_profiles")
    profile = models.ForeignKey(Profile, on_delete=models.PROTECT, related_name="member_profiles")
    scale_factor = models.FloatField(default=1.0, validators=[MinValueValidator(0.0)])

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