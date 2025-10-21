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

    # Global admin-ish on main page
    path("profiles/create/", v.profile_create, name="profile_create"),
    path("global-params/create/", v.global_parameter_create, name="global_parameter_create"),

    # CSV template
    path("download/timeseries-template.csv", v.csv_template_timeseries, name="csv_template_timeseries"),
    
    # URLs for tabs in General Data section
    path("tabs/profiles/", v.profiles_tab, name="profiles_tab"),
    path("tabs/global-parameters/", v.global_parameters_tab, name="global_parameters_tab"),
    path("projects/<int:project_id>/analysis/", v.project_analysis_page, name="project_analysis_page"),

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)