from django.contrib import admin
from .models import (
    Project,
    Member,
    Dataset,
    GlobalParameter,
    StageTwoScenario,
    StageThreeScenario,
)

@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("name", "phase", "created_at")
    search_fields = ("name",)

@admin.register(Member)
class MemberAdmin(admin.ModelAdmin):
    list_display = ("name", "project", "dataset")
    list_filter = ("project", "dataset")

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("name", "created_at")
    search_fields = ("name",)

@admin.register(GlobalParameter)
class GlobalParameterAdmin(admin.ModelAdmin):
    list_display = ("key", "updated_at")
    search_fields = ("key",)


@admin.register(StageTwoScenario)
class StageTwoScenarioAdmin(admin.ModelAdmin):
    list_display = ("name", "project", "created_at", "updated_at")
    list_filter = ("project",)
    search_fields = ("name", "project__name")


@admin.register(StageThreeScenario)
class StageThreeScenarioAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "project",
        "community_price_eur_per_kwh",
        "price_min_eur_per_kwh",
        "price_max_eur_per_kwh",
    )
    list_filter = ("project",)
    search_fields = ("name", "project__name")