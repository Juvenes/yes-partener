from django.contrib import admin
from django.urls import path
from core import views as v
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", v.project_list, name="project_list"),
    path("projects/create/", v.project_create, name="project_create"),
    path("projects/<int:project_id>/", v.project_detail, name="project_detail"),
    path("projects/<int:project_id>/members/create/", v.member_create, name="member_create"),
    path("projects/<int:project_id>/analysis/", v.project_analysis_page, name="project_analysis_page"),

    # New dedicated pages
    path("profiles/", v.profile_list, name="profiles"),
    path("global-parameters/", v.global_parameter_list, name="global_parameters"),

    # Keep create views, but they will redirect differently
    path("profiles/create/", v.profile_create, name="profile_create"),
    path("global-params/create/", v.global_parameter_create, name="global_parameter_create"),

    # CSV template
    path("download/timeseries-template.csv", v.csv_template_timeseries, name="csv_template_timeseries"),

    # Remove old tab URLs
    # path("tabs/profiles/", v.profiles_tab, name="profiles_tab"),
    # path("tabs/global-parameters/", v.global_parameters_tab, name="global_parameters_tab"),

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)