#from tkinter.filedialog import Open
from ast import Subscript
from django.shortcuts import render
from django.http import HttpResponse
from datetime import datetime
from django.http import Http404
from baekho.models import ChinaModel, VtModel, Country, Opening
import csv
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup as bs
from tqdm import tqdm
import time
import schedule 
import datetime
from django.conf import settings
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import googletrans
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
    
    now = datetime.datetime.now()
    print("======현재시간:",now,"=====")
    
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
        CSV_PATH = vietnam()
        # CSV_PATH = "vietnam.csv"

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

    return render(request, 'warn/detail.html', context=context)

def vietnam():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument("--single-process")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome('/home/minhee/steve/baekho/chromedriver',chrome_options=chrome_options)

    # driver = webdriver.Chrome(chrome_driver_path, options=options)

    base_url = 'https://www.customs.gov.vn' #합격
    # https://pythondocs.net/selenium/%EC%85%80%EB%A0%88%EB%8B%88%EC%9B%80-wait-%EA%B0%9C%EB%85%90-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-implicitly-wait-vs-explicitly-wait/
    # 타입슬립 종류 주는법

    try:
        driver.get(base_url+'/index.jsp?pageId=4&cid=30')
        driver.maximize_window()
        WebDriverWait(driver, 200).until(EC.presence_of_element_located((By.XPATH, "/html/body/div/div[2]/section[2]/div/div[1]/div[2]/div[1]/div[1]/div/module/div[1]")))
    except Exception as e:
        print(e)

    html = driver.page_source
    soup = bs(html, 'html.parser')
    list = soup.find('div', 'content-list module-content content-yellow content-page').find_all('div','content_item')

    title2, bdate2, contents2, url2 = [],[],[], []

    for i in tqdm(list[0:3]):
        new_url = base_url + i.find('a')['href']
        driver.get(new_url)
        WebDriverWait(driver, 200).until(EC.presence_of_element_located((By.XPATH, "/html/body/div/div[2]/section[2]/div/div[1]/div[2]/div[1]/div[1]/div/module/div[2]/h1")))
        # river.implicitly_wait(100)
        page_html = driver.page_source
        page_soup = bs(page_html, 'html.parser')
        
        title_1 = page_soup.find('div','hot-news').find('h1','component-title')  
        title = title_1.get_text()                 # 게시물 제목
        title2.append(title)                       # 게시물 제목 리스트에 추가

        bdate = page_soup.find('div','hot-news').find('content_header').get_text( )  # 작성일자
        bdate2.append(bdate)                       # 작성일자 리스트에 추가

        contents = page_soup.find('div','hot-news').find('content','component-content').get_text( )   # 게시물 요약 내용
        contents2.append(contents)                 # 게시물 내용 리스트에 추가

        url = driver.current_url
        url2.append(url)
        
    rows=[]
    nums= [num for num in range(1, len(title2)+1)]
    for n, d, t, c, u in zip(nums, bdate2, title2, contents2, url2):
        row=[]
        row.append(n)
        row.append(d)
        row.append(t)
        row.append(c)
        row.append(u)
        rows.append(row)

    data = pd.DataFrame(rows, columns=['번호','날짜','제목','내용','주소'])
    #path = "vietnam.csv"
    #data.to_csv(path, index=False)

    ##############################

    translator = googletrans.Translator()
    #vietnam_kor = pd.read_csv('vietnam.csv', encoding="utf8")
    #vietnam_eng = pd.read_csv('vietnam.csv', encoding="utf8")

    result1, result2 = [], []

    result1=[]
    nums= [num for num in range(1, len(title2)+1)]
    for n, d, t, c, u in zip(nums, bdate2, title2, contents2, url2):
        result=[]

        result.append(n)
        result.append(d)

        # result.append(t)
        word = translator.translate(t, src='vi',dest='ko') #en
        result.append(word.text)

        result.append(c)
        result.append(u)
        result1.append(result)

    # for i in title2:
    #     result = translator.translate(i, src='vi',dest='ko') #en
    #     result1.append(result.text)
    
    data = pd.DataFrame(result1, columns=['번호','날짜','제목','내용','주소'])
    path = "vietnam_kor.csv"
    data.to_csv(path, index=False)

    return path