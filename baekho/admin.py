from django.contrib import admin
from baekho.models import ChinaModel, USAModel, JapanModel, VtModel, AuModel, Country,Opening
# Register your models here.

admin.site.register(Country)
admin.site.register(Opening)

admin.site.register(ChinaModel)
admin.site.register(USAModel)
admin.site.register(JapanModel)
admin.site.register(VtModel)
admin.site.register(AuModel)
