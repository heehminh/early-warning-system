from django.shortcuts import render
from django.http import HttpResponse
from datetime import datetime
from django.http import Http404
from baekho.models import Country
# Create your views here.

def index(request):
    context = dict()
    #country = Country.objects.filter(id=pk)
    countries = Country.objects.all()
    context["countries"] = countries
    return render(request, 'warn/index.html', context=context)

def baekho_detail(request, pk):
    context = dict()
    country = Country.objects.get(id=pk)
    context["country"] = country
    return render(request, 'warn/detail.html', context=context)