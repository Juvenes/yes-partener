from django.contrib import admin
from .models import Project, Member, Profile, MemberProfile, GlobalParameter

class MemberProfileInline(admin.TabularInline):
    model = MemberProfile
    extra = 0

@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("name", "phase", "created_at")
    search_fields = ("name",)

@admin.register(Member)
class MemberAdmin(admin.ModelAdmin):
    list_display = ("name", "project")
    list_filter = ("project",)
    inlines = [MemberProfileInline]

@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ("name", "profile_type", "version", "is_active")
    list_filter = ("profile_type", "is_active",)
    search_fields = ("name",)

@admin.register(GlobalParameter)
class GlobalParameterAdmin(admin.ModelAdmin):
    list_display = ("key", "updated_at")
    search_fields = ("key",)