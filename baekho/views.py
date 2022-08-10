from tkinter.filedialog import Open
from django.shortcuts import render
from django.http import HttpResponse
from datetime import datetime
from django.http import Http404
from baekho.models import ChinaModel, VtModel, Country, Opening
import csv
# Create your views here.

# def index(request):
#     context = dict()
#     #country = Country.objects.filter(id=pk)
#     countries = Country.objects.all()
#     context["countries"] = countries
#     return render(request, 'warn/index.html', context=context)

def baekho_detail(request, pk):
    context = dict()
    country = Country.objects.get(id=pk)
    context["country"] = country

    countries = Country.objects.all()
    context["countries"] = countries

    # csv 파일 열기
    # 1) csv 파일 불러와서 Opening model에 저장하기
    # 2) Opening model에서 title, url 불러오기

    if (pk==1): # 중국
        CSV_PATH = "baekho/static/warn/csv/data_china.csv"

        records = ChinaModel.objects.all()
        records.delete()

        file = open(CSV_PATH, encoding="utf8")
        reader = csv.reader(file)
        next(reader, None)
        list = []
        for row in reader:
            list.append(ChinaModel(title=row[2],
                                date=row[1], 
                                url=row[4]))
        ChinaModel.objects.bulk_create(list)

        opening = ChinaModel.objects.all()
        context["opening"] = opening


    elif (pk==4): # 베트남 
        CSV_PATH = "baekho/static/warn/csv/vietnam.csv"

        records = VtModel.objects.all()
        records.delete()

        file = open(CSV_PATH, encoding="utf8")
        reader = csv.reader(file)
        next(reader, None)
        list = []
        for row in reader:
            list.append(VtModel(title=row[2],
                                date=row[1], 
                                url=row[4]))
        VtModel.objects.bulk_create(list)

        opening = VtModel.objects.all()
        context["opening"] = opening




    

    # 처음 Model에 저장할 때 필요한 코드 -> file~ Opening.objects.bulk_create(list) 1회 실행
    
    # records = Opening.objects.all()
    # records.delete()

    # file = open(CSV_PATH, encoding="utf8")
    # reader = csv.reader(file)
    # next(reader, None)
    # list = []
    # for row in reader:
    #     list.append(Opening(title=row[2],
    #                          date="2022-08-10", 
    #                         url=" "))
    # Opening.objects.bulk_create(list)

    # opening = Opening.objects.all()
    # context["opening"] = opening
    
    return render(request, 'warn/detail.html', context=context)