from django.contrib import admin
from baekho.models import Country,Opening, HeadOffice
# Register your models here.

admin.site.register(Country)
admin.site.register(Opening)

# admin.site.register(ChinaModepl)
# admin.site.register(USAModel)
# admin.site.register(JapanModel)
# admin.site.register(VtModel)
# admin.site.register(AuModel)

admin.site.register(HeadOffice)