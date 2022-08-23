from django.apps import AppConfig
from django.conf import settings

class BaekhoConfig(AppConfig):
    name = 'baekho'

    def ready(self):
        from baekho import operator
        operator.start()
