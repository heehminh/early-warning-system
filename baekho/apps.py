from django.apps import AppConfig
from django.conf import settings

class BaekhoConfig(AppConfig):
    name = 'baekho'

    def ready(self):
        from . import operator
        operator.start()
