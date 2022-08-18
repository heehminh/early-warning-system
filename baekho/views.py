#from tkinter.filedialog import Open
from ast import Subscript
from asyncio import Handle
from itertools import count
import numbers
#from curses.ascii import VT
from django.shortcuts import render
from django.http import HttpResponse
from datetime import datetime
from django.http import Http404
from baekho.models import Country, Opening, HeadOffice
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
from googletrans import Translator
from sentence_transformers import SentenceTransformer
import re
import numpy as np
from numpy import dot
from numpy.linalg import norm
import nltk
from nltk.corpus import stopwords

import requests
import warnings
import pickle
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import os

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from konlpy.tag import Okt
from PIL import Image
from keybert import KeyBERT

def update_csv():
    # 원래 csv 삭제하고 새로 
    print("==스케줄러 시작==")
    today = datetime.date.today()

    # vietnam()
    print("==스케줄러 완료==")

def baekho_detail(request, pk):

    context = dict()
    country = Country.objects.get(id=pk)
    context["country"] = country

    countries = Country.objects.all()
    context["countries"] = countries
    now = datetime.datetime.now()
    print("======현재시간:",now,"=====")

    def make_context(CSV_PATH, name):
            records = HeadOffice.objects.all()
            records.delete()

            file = open(CSV_PATH, encoding="utf8")
            reader = csv.reader(file)
            next(reader, None)
            list = []
            cnt = 0
            for row in reader:
                cnt += 1
                
                print(row[0]+"번째 출력\n")

                ngram_list = row[7].split("||")
                code_list = row[8].split("||")
                name_list = row[9].split("||")
                sim_list = row[10].split("||")

                print("==데이터전처리전==")        
                print("ngram_list:",ngram_list)
                print("code_list:",code_list)
                print("name_list:",name_list)
                print("sim_list:",sim_list)

                if len(ngram_list) == 1:
                    ngram_list[0] = "NaN"
                    code_list[0] = "NaN"
                    name_list[0] = "NaN"
                    sim_list[0] = "NaN"

                    for i in range(3):
                        ngram_list.append(" ")
                        code_list.append(" ")
                        name_list.append(" ")
                        sim_list.append(" ")
                

                elif len(ngram_list) == 2:
                    ngram_list[1] = ""
                    code_list[1] = ""
                    name_list[1] = ""
                    sim_list[1] = ""

                    for i in range(2):
                        ngram_list.append(" ")
                        code_list.append(" ")
                        name_list.append(" ")
                        sim_list.append(" ")

                elif len(ngram_list) == 3:
                    ngram_list[2] = ""
                    code_list[2] = ""
                    name_list[2] = ""
                    sim_list[2] = ""

                    ngram_list.append(" ")
                    code_list.append(" ")
                    name_list.append(" ")
                    sim_list.append(" ")
                        
                elif len(ngram_list) == 4:
                    ngram_list[3] = ""
                    code_list[3] = ""
                    name_list[3] = ""
                    sim_list[3] = ""

                print("==데이터전처리후==")
                print("ngram_list:",ngram_list)
                print("code_list:",code_list)
                print("name_list:",name_list)
                print("sim_list:",sim_list) 
                

                list.append(HeadOffice(
                                    number = cnt,
                                    title=row[2],
                                    date=row[3], 
                                    url=row[12],
                                    img="/static/warn/image/"+str(name)+"-au"+str(cnt)+".png",
                                    country=name,
                                    
                                    word1_ngram=ngram_list[0], 
                                    word1_code=code_list[0],
                                    word1_name=name_list[0],
                                    word1_sim=sim_list[0],

                                    word2_ngram=ngram_list[1], 
                                    word2_code=code_list[1],
                                    word2_name=name_list[1],
                                    word2_sim=sim_list[1],

                                    word3_ngram=ngram_list[2], 
                                    word3_code=code_list[2],
                                    word3_name=name_list[2],
                                    word3_sim=sim_list[2],

                                    word4_ngram=ngram_list[3], 
                                    word4_code=code_list[3],
                                    word4_name=name_list[3],
                                    word4_sim=sim_list[3],
        
                                    ))

        
                                    
            HeadOffice.objects.bulk_create(list)

            opening = HeadOffice.objects.all()
            context["opening"] = opening

            return 
    
    if (pk==1): # 중국
        #CSV_PATH = china()
        CSV_PATH = "../baekho/result_cn.csv" 
        country = "중국"

    elif (pk==2): # 미국
        CSV_PATH = usa()
        # CSV_PATH = "result_cn.csv.part"
        country = "미국"
        
    elif (pk==3): # 일본
        CSV_PATH = japan()
        # CSV_PATH = 
        country = "일본"

    elif (pk==4): # 베트남 
        # CSV_PATH = vietnam()
        CSV_PATH = "../baekho/result_vi.csv"   
        country = "베트남" 

    elif (pk==5): # 호주 
        # CSV_PATH = "../baekho/"/australia()
        CSV_PATH = "../baekho/result_au.csv"
        country = "호주"
    
    make_context(CSV_PATH, country)

    return render(request, 'warn/detail.html', context=context)

def subpage(request, pk, title):

    context = dict() 
    info = HeadOffice.objects.get(number=title)
    context["info"] = info

    return render(request, "warn/subpage.html", context=context)


def vietnam():
    nltk.download('stopwords')
    origin_sw = stopwords.words('english')
    # cn_sw = origin_sw + ['china','chinese','customs', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    vt_sw = origin_sw + ['deputy', 'documents','minister','committee','committe','nguyen','vietnam','customs', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    data = pd.read_excel('../hs_vt_eng.xlsx')
    data['hs_embedding'] = data.apply(lambda row: model.encode(row.subject), axis = 1) #hs품목

    ##############################################################
    ## 함수 정의
    ##############################################################

    def cos_sim(A, B):
        return dot(A, B)/(norm(A)*norm(B))

    def get_text(soup):
        return soup.get_text().replace('\n', ' ').replace('\u3000','').strip()

    def rid_eng(txt):
        ret = re.sub("[A-Za-z]", "", txt)
        ret = re.sub("  ", "", ret)
        return ret

    def get_score(ngrams, data, thresh=0.5): #thresh(유사도점수) 기준치는 외부함수에서 덮어써진다
        rows=[]
        not_none = False
        for ngram in ngrams:
            old, new = 0, 0
            tmp = model.encode(ngram) #영어내용 단어 별보 백터값 추출
            for a,b,c in zip(data['hs_embedding'], data['subject'], data['number']):
                row= []
                new = cos_sim(a, tmp)
                if old < new and new > thresh:
                    old = new 
                    row.append(ngram)
                    row.append(str(c)) #hscode
                    row.append(b) #hs품목
                    row.append(new) #코사인 유사도
                rows.append(row)
        for r in rows:
            if r:
                not_none = True
        if not_none:
            df = pd.DataFrame(rows, columns=['ngram','code', 'name', 'sim'])
            df = df.astype('str')
            df = df.sort_values(by='sim', ascending=False) #유사도 기준 내림차순
            df = df.drop_duplicates(['name'], keep = 'first') #첫 데이터 뺴고 중복 제거
            df = df.drop_duplicates(['ngram'], keep = 'first')
            df = df.head(5)
            ret = df.drop(0, axis=0) #상위 5개 추출이 안될때 NaN제거해줌
        else:
            ret = pd.DataFrame([[None,None,None,None]],columns=['ngram','code', 'name', 'sim'])
        return ret

    def rid_sc(txt):
        return re.sub(r"[^a-zA-Z0-9 ]","",txt).lower()
        
    def get_hscode(txt):
        p = re.compile('[0-9]{8}')
        result = p.findall(txt)
        s = set(result)
        return list(s)
        
    def generate_N_grams(text,ngram, stopwords):
        words=[word for word in text.replace("\n", "").split(" ") if word not in set(stopwords)] #영어전체내용 불용어제거 후 단어별로 나눔
        temp=zip(*[words[i:] for i in range(0,ngram)])
        ans=[' '.join(ngram) for ngram in temp]
        return ans

    def get_hscode(txt):
        p = re.compile('[0-9]{4}.[0-9]{2}')
        result = p.findall(txt)
        s = set(result)
        return list(s)

    def make_code(lis):
        ls = lis.copy()
        for i, txt in enumerate(ls):
            if ('.' in txt):
                tmp = txt.replace('.', '')
                ls[i] = tmp
            else:
                ls[i] = 'none'
        while True:
            if 'none' not in ls:
                break
            else:
                ls.remove('none')
        return ls

    def make_dates_vi(txt):
        splited_txt = txt.split(' ')[0].split('/')
        ret=''
        if splited_txt[0].startswith('0'):
            splited_txt[0] = splited_txt[0][1:]

        if splited_txt[1].startswith('0'):
            splited_txt[1] = splited_txt[1][1:]

        ret += splited_txt[2] + '년 ' + splited_txt[1] + '월 ' + splited_txt[0] + "일 게시"
        return ret

    ##############################################################
    ## 크롤링 파트
    ##############################################################

    #chrome_driver_path = './chromedriver.exe'
    chrome_driver_path = '../baekho/chromedriver.exe'
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument("disable-blink-features=AutomationControlled") #selenium 엿먹이는 코드 다시 엿먹이기
    options.add_argument('headless') #팝업창 안뜨게 제어
    options.add_argument('window-size=1920x1080')
    #options.add_argument("disable-gpu")
    args = ["hide_console", ]
    #options.add_argument("--disable-extensions");
    options.add_argument("disable-infobars");
    options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(chrome_driver_path, options=options, service_args=args)

    base_url = 'https://www.customs.gov.vn/' 

    titles, dates, contents, links = [], [], [], []


    print("======베트남 크롤링 시작=====")
    driver.get(base_url+'/index.jsp?pageId=4&cid=30')
    #time.sleep(5)
    driver.maximize_window()
    WebDriverWait(driver, 200).until(EC.presence_of_element_located((By.XPATH, "/html/body/div/div[2]/section[2]/div/div[1]/div[2]/div[1]/div[1]/div/module/div[1]")))
    html = driver.page_source
    soup = bs(html, 'html.parser')
    lists = soup.find('div', id='listItems').find_all('div','content_item')[:5]

    for l in lists[0:5]:
        new_url = base_url + l.find('a')['href']
        link = new_url
        driver.get(new_url)
        WebDriverWait(driver, 200).until(EC.presence_of_element_located((By.XPATH, "/html/body/div/div[2]/section[2]/div/div[1]/div[2]/div[1]/div[1]/div/module/div[2]/h1")))
        #time.sleep(5)

        li_html = driver.page_source
        li_soup = bs(li_html, 'html.parser')

        cont = li_soup.find('content', 'component-content')
        joined_text = get_text(cont)

        title = get_text(li_soup.find('h1','component-title'))
        date_tmp = get_text(l.find('p','note-layout'))
        date = make_dates_vi(date_tmp)
        titles.append(title)
        dates.append(date)
        contents.append(joined_text)
        links.append(link)

    ##############################################################
    ## 번역 파트
    ##############################################################

    print("======번역 시작=====")  
    translator = Translator()
    titles_tr, contents_tr= [],[]

    for content, title in zip(contents, titles):
        if len(content) > 4000:
            c1 = content[:4000]
            c2 = content[4001:]
            c1_tmp = translator.translate(c1,src='vi', dest="en" )
            c2_tmp = translator.translate(c2,src='vi', dest="en" )
            cont_tmp = c1_tmp.text + c2_tmp.text
            cont_translated = rid_sc(cont_tmp)
            
        else:
            cont_tmp = translator.translate(content ,src='vi', dest="en")
            cont_translated = rid_sc(cont_tmp.text)

        title_tmp = translator.translate(title, src='vi', dest="ko")
        title_translated = title_tmp.text
        contents_tr.append(cont_translated)
        titles_tr.append(title_translated)

    # from transformers import pipeline

    # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    ##############################################################
    ## 키버트 키워드 요약 파트
    ##############################################################
    print("======키워드 요약 시작=====")
    from keybert import KeyBERT

    keywords_list = []
    kw_model = KeyBERT()
    for t in contents_tr:
        k = []
        p = re.sub(r"[^a-zA-Z ]","",t) #영어랑 띄어쓰기만 살리고 나머지 특수기호 등 다 제거
        keywords = kw_model.extract_keywords(p, keyphrase_ngram_range=(3,3), stop_words=set(vt_sw), top_n=10, diversity=0.9) #트라이그램 유사도 상위 10개 품목
        for keyword in keywords:
            k.append(keyword[0])
        keywords_list.append(k)

    ##############################################################
    ## 키워드 hscode 비교 파트
    ##############################################################

    print("======키워드 hscode 비교 시작=====")
    ngrams, codes, names, sims = [],[],[],[] #ngrams: 3개 단어, codes:hs코드, names:hs품목명, sim:유사도
    for c in contents_tr:
        txt = generate_N_grams(c,2,vt_sw) ##영어전체내용 불용어제거 후 단어별로 나눔
        ls = get_score(txt, data, 0.6)
        a,b,c,d = '','','',''
        for ng, co, na, si in zip(ls.ngram,ls.code,ls.name,ls.sim):
            if ng:
                a += ng + '||'
                if len(co) == 3:
                    co = '0' + co
                b += co + '||'        
                c += na + '||'        
                d += si + '||'  
            else:
                a,b,c,d = ' ',' ',' ',' '
        while True:
            if 'None||' in a:
                a = a.strip('None||')
                b = b.strip('None||')
                c = c.strip('None||')
                d = d.strip('an||')
            else:
                break
        ngrams.append(a)
        codes.append(b)    
        names.append(c)    
        sims.append(d)

    ##############################################################
    ##찐 hscode 검출 파트
    ##############################################################
    print("======hscode 검출 시작=====")
    real_codes = []
    for v in contents:
        ls = make_code(get_hscode(v))
        if len(ls) == 0:
            ls.append('None')
        real_codes.append(ls)

    ##############################################################
    ##결과물 만들기 파트
    ##############################################################
    print("======csv 저장=====")
    nums = range(1, len(titles)+1)
    rows = []
    for a,b,c,d,e,f,g,h,i,j,k,l,m in zip(nums, titles, titles_tr ,dates, contents, contents_tr, keywords_list, ngrams, codes, names, sims, real_codes, links):
        row= [a,b,c,d,e,f,g,h,i,j,k,l,m]
        rows.append(row)

    result = pd.DataFrame(rows, columns=['번호','제목','제목(한국어번역)','날짜','내용','내용(영어번역)','키워드 요약','일치 키워드','hs코드','품목','유사도','본문에 진짜 코드','링크'])
    path = "result_vi.csv"
    result.to_csv('./result_vi.csv', index=False)
    print("======베트남 크롤링 완료=====")

    ##############################################################
    ##############################################################
    ##############################################################
    return path

def china():

    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.chrome.service import Service
    from bs4 import BeautifulSoup as bs
    from googletrans import Translator
    from sentence_transformers import SentenceTransformer
    import re
    import pandas as pd
    import numpy as np
    from numpy import dot
    from numpy.linalg import norm
    import nltk
    from nltk.corpus import stopwords

    ##############################################################
    ## 필요한 것들 미리 정의
    ##############################################################

    nltk.download('stopwords')
    origin_sw = stopwords.words('english')
    # cn_sw = origin_sw + ['china','chinese','customs', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    vt_sw = origin_sw + ['deputy', 'documents','minister','committee','committe','nguyen','vietnam','customs', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    data1 = pd.read_excel('../hs_china.xlsx')
    data = data1.loc[:,"number":"subject"]
    data['hs_embedding'] = data.apply(lambda row: model.encode(row.subject), axis = 1) #hs품목 
    #data = pd.read_excel('../hscode/hscode.xlsx')

    ##############################################################
    ## 함수 정의
    ##############################################################

    def cos_sim(A, B):
        return dot(A, B)/(norm(A)*norm(B))

    def get_text(soup):
        return soup.get_text().replace('\n', ' ').replace('\n',' ').strip() #replace('\u3000','')

    def rid_eng(txt):
        ret = re.sub("[A-Za-z]", "", txt)
        ret = re.sub("  ", "", ret)
        return ret

    def get_score(ngrams, data, thresh=0.5): #thresh(유사도점수) 기준치는 외부함수에서 덮어써진다
        rows=[]
        not_none = False
        for ngram in ngrams:
            old, new = 0, 0
            tmp = model.encode(ngram) #영어내용 단어 별보 백터값 추출
            for a,b,c in zip(data['hs_embedding'], data['subject'], data['number']):
                row= []
                new = cos_sim(a, tmp)
                if old < new and new > thresh:
                    old = new 
                    row.append(ngram)
                    row.append(str(c)) #hscode
                    row.append(b) #hs품목
                    row.append(new) #코사인 유사도
                rows.append(row)
        for r in rows:
            if r:
                not_none = True
        if not_none:
            df = pd.DataFrame(rows, columns=['ngram','code', 'name', 'sim'])
            df = df.astype('str')
            df = df.sort_values(by='sim', ascending=False) #유사도 기준 내림차순
            df = df.drop_duplicates(['name'], keep = 'first') #첫 데이터 뺴고 중복 제거
            df = df.drop_duplicates(['ngram'], keep = 'first')
            df = df.head(5)
            ret = df.drop(0, axis=0) #상위 5개 추출이 안될때 NaN제거해줌
            #ret = df.dropna() #상위 5개 추출이 안될때 NaN제거해줌
        else:
            ret = pd.DataFrame([[None,None,None,None]],columns=['ngram','code', 'name', 'sim'])
        return ret

    def rid_sc(txt):
        return re.sub(r"[^a-zA-Z0-9 ]","",txt).lower()
        
    def get_hscode(txt):
        p = re.compile('[0-9]{8}')
        result = p.findall(txt)
        s = set(result)
        return list(s)
        
    def generate_N_grams(text,ngram, stopwords):
        words=[word for word in text.replace("\n", "").split(" ") if word not in set(stopwords)] #영어전체내용 불용어제거 후 단어별로 나눔
        temp=zip(*[words[i:] for i in range(0,ngram)])
        ans=[' '.join(ngram) for ngram in temp]
        return ans

    def get_hscode(txt):
        p = re.compile('[0-9]{4}.[0-9]{2}')
        result = p.findall(txt)
        s = set(result)
        return list(s)

    def make_code(lis):
        ls = lis.copy()
        for i, txt in enumerate(ls):
            if ('.' in txt):
                tmp = txt.replace('.', '')
                ls[i] = tmp
            else:
                ls[i] = 'none'
        while True:
            if 'none' not in ls:
                break
            else:
                ls.remove('none')
        return ls
    
    def make_dates_cn(txt):
        splited_txt = txt.split('-')
        ret=''
        if splited_txt[1].startswith('0'):
            splited_txt[1] = splited_txt[1][1:]

        if splited_txt[2].startswith('0'):
            splited_txt[2] = splited_txt[2][1:]

        ret += splited_txt[0] + '년 ' + splited_txt[1] + '월 ' + splited_txt[2] + "일 게시"
        return ret

    ##############################################################
    ## 크롤링 파트
    ##############################################################
    print("크롤링파트 진입")
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import time
    #chrome_driver_path = './chromedriver.exe'
    chrome_driver_path = '../baekho/chromedriver.exe'
    options = webdriver.ChromeOptions()

    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument("disable-blink-features=AutomationControlled") #selenium 엿먹이는 코드 다시 엿먹이기
    #options.add_argument('headless') #팝업창 안뜨게 제어
    options.add_argument('window-size=1920x1080')
    #options.add_argument("disable-gpu")
    args = ["hide_console", ]
    #options.add_argument("--disable-extensions");
    options.add_argument("disable-infobars")
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(chrome_driver_path, options=options)

    base_url = 'http://www.customs.gov.cn' 

    titles, dates, contents, links, files = [], [], [], [], []

    driver.get(base_url+'/eportal/ui?pageId=2480148&currentPage=1&moduleId=08654b53d1cc42478813b0b2feddcb57&staticRequest=yes')
    time.sleep(3)
    driver.maximize_window()
    WebDriverWait(driver, 200).until(EC.presence_of_element_located((By.XPATH, "/html/body/div[4]/div/div[2]/div/div[2]")))
    html = driver.page_source
    soup = bs(html, 'html.parser')
    lists = soup.find('ul', 'conList_ull').find_all('li')

    from tqdm import tqdm

    print("크롤링 시작")
    for l in tqdm(lists[0:5]):
        title = l.find('a')['title']
        date_tmp = get_text(l.find('span'))
        date = make_dates_cn(date_tmp)

        new_url = base_url + l.find('a')['href']
        link = new_url
        driver.get(new_url)
        WebDriverWait(driver, 200).until(EC.presence_of_element_located((By.XPATH, "/html/body/div[4]/div[1]/div[2]/div/div")))

        li_html = driver.page_source
        li_soup = bs(li_html, 'html.parser')

        ###########################
        cont = li_soup.find('div', 'easysite-news-text')
        joined_text = get_text(cont)
        #################

        titles.append(title)
        dates.append(date)
        contents.append(joined_text)
        links.append(link)
    
    driver.close()
        
    ##############################################################
    ## 번역 파트
    ##############################################################
    print("번역 시작")
    translator = Translator()
    titles_tr, contents_tr= [],[]

    for content, title in zip(contents, titles):
        if len(content) > 4000:
            c1 = content[:4000]
            c2 = content[4001:]
            c1_tmp = translator.translate(c1,src='zh-cn', dest="en" )
            c2_tmp = translator.translate(c2,src='zh-cn', dest="en" )
            cont_tmp = c1_tmp.text + c2_tmp.text
            cont_translated = rid_sc(cont_tmp)  
        else:
            cont_tmp = translator.translate(content ,src='zh-cn', dest="en")
            cont_translated = rid_sc(cont_tmp.text)
        title_tmp = translator.translate(title, src='zh-cn', dest="ko")
        title_translated = title_tmp.text
        contents_tr.append(cont_translated)
        titles_tr.append(title_translated)

    # from transformers import pipeline

    # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    ##############################################################
    ## 키버트 키워드 요약 파트
    ##############################################################

    from keybert import KeyBERT

    keywords_list = []
    kw_model = KeyBERT()
    for t in contents_tr:
        k = []
        p = re.sub(r"[^a-zA-Z ]","",t) #영어랑 띄어쓰기만 살리고 나머지 특수기호 등 다 제거
        keywords = kw_model.extract_keywords(p, keyphrase_ngram_range=(3,3), stop_words=set(vt_sw), top_n=10, diversity=0.9, use_mmr=True) #트라이그램 유사도 상위 10개 품목
        for keyword in keywords:
            k.append(keyword[0])
        keywords_list.append(k)

    ##############################################################
    ## 키워드 hscode 비교 파트
    ##############################################################

    ngrams, codes, names, sims = [],[],[],[] #ngrams: 3개 단어, codes:hs코드, names:hs품목명, sim:유사도
    for c in contents_tr:
        txt = generate_N_grams(c,2,vt_sw) ##영어전체내용 불용어제거 후 단어별로 나눔
        ls = get_score(txt, data, 0.6)
        a,b,c,d = '','','',''
        for ng, co, na, si in zip(ls.ngram,ls.code,ls.name,ls.sim):
            if ng:
                a += ng + '||'
                if len(co) == 3:
                    co = '0' + co
                b += co + '||'        
                c += na + '||'        
                d += si + '||'  
            else:
                a,b,c,d = ' ',' ',' ',' '

        while True:
            if 'None||' in a:
                a = a.strip('None||')
                b = b.strip('None||')
                c = c.strip('None||')
                d = d.strip('an||')
            else:
                break
        a += '|| '
        b += '|| '
        c += '|| '
        d += '|| '

        ngrams.append(a)
        codes.append(b)
        names.append(c)
        sims.append(d)

    #ngrams.append(a)
    #ngrams.append('/ ')
    #codes.append(b)
    #codes.append('/ ')    
    #names.append(c)
    #names.append('/ ')    
    #sims.append(d)
    #sims.append('/ ')

    ##############################################################
    ##찐 hscode 검출 파트
    ##############################################################

    real_codes = []
    for v in contents:
        ls = make_code(get_hscode(v))
        if len(ls) == 0:
            ls.append('None')
        real_codes.append(ls)

    ##############################################################
    ##결과물 만들기 파트
    ##############################################################

    nums = range(1, len(titles)+1)
    rows = []
    for a,b,c,d,e,f,g,h,i,j,k,l, m in zip(nums, titles, titles_tr ,dates, contents, contents_tr, keywords_list, ngrams, codes, names, sims, real_codes, links):
        row= [a,b,c,d,e,f,g,h,i,j,k,l,m]
        rows.append(row)

    result = pd.DataFrame(rows, columns=['번호','제목','제목(한국어번역)','날짜','내용','내용(영어번역)','키워드 요약','일치 키워드','hs코드','품목','유사도','본문에 진짜 코드','링크'])

    result.to_csv('./result_cn.csv', index=False)

    print('done')

def japan():
    return 

def australia():
    warnings.filterwarnings('ignore')

    ######################################################
    ### 필요한 것들 정의
    ######################################################
    nltk.download('stopwords')
    origin_sw = stopwords.words('english')
    # cn_sw = origin_sw + ['china','chinese','customs', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    #vt_sw = origin_sw + ['deputy', 'documents','minister','committee','committe','nguyen','vietnam','customs', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    au_sw = origin_sw + ['australian','january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # data = pd.read_excel('../hscode/hscode.xlsx')
    # data['hs_embedding'] = data.apply(lambda row: model.encode(row.SUBJECT), axis = 1)

    # data의 pkl화 된 데이터프레임 미리 로딩
    # 아래 path를 바꿔주면 거기서 로드할수 있음
    with open('hs_embedded.pkl', 'rb') as f: #개체의 특성을 피클모델이 이용할 수 있는 객체를 담아낸것
        data = pickle.load(f)
    
    ######################################################
    ### 함수 정의
    ######################################################


    # 코사인 유사도 계산
    import enum

    def cos_sim(A, B):
        return dot(A, B)/(norm(A)*norm(B))

    # beautiful_soup 텍스트 추출
    def get_text(soup):
        return soup.get_text().replace('\n', ' ').replace('\n',' ').strip() #replace('\u3000','')

    # ngram과 data 유사도 추출
    def get_score(ngrams, data, thresh=0.5):
        rows=[]
        not_none = False
        for ngram in ngrams:
            old, new = 0, 0
            tmp = model.encode(ngram)
            for a,b,c in zip(data['hs_embedding'], data['SUBJECT'], data['HSCODE']):
                row= []
                new = cos_sim(a, tmp)
                if old < new and new > thresh:
                    old = new
                    row.append(ngram)
                    row.append(str(c))
                    row.append(b)
                    row.append(new)
                rows.append(row)
        for r in rows:
            if r:
                not_none = True
        if not_none:
            df = pd.DataFrame(rows, columns=['ngram','code', 'name', 'sim'])
            df = df.astype('str')
            df = df.sort_values(by='sim', ascending=False)
            df = df.drop_duplicates(['name'], keep = 'first')
            df = df.drop_duplicates(['ngram'], keep = 'first')
            df = df.head(5)
            ret = df.drop(0, axis=0)
        else:
            ret = pd.DataFrame([[None,None,None,None]],columns=['ngram','code', 'name', 'sim'])
        return ret

    # 특수문자 제거
    def rid_sc(txt):
        ret = re.sub(r"[^a-zA-Z ]"," ",txt).lower()
        ret = ' '.join(ret.split())
        return ret
        
        
    # ngram 생성기
    def generate_N_grams(text,ngram, stopwords):
        words=[word for word in text.replace("\n", "").split(" ") if word not in set(stopwords)]  
        temp=zip(*[words[i:] for i in range(0,ngram)])
        ans=[' '.join(ngram) for ngram in temp]
        while True:
            if ' ' in ans:
                ans.remove(' ')
            else:
                break
        return ans

    # hs-code 검출 (정규표현식 이용)
    def get_hscode(txt):
        p = re.compile('[0-9]{4}[.][0-9]{2}')
        result = p.findall(txt)
        for i,r in enumerate(result):
            r = r.replace('.','')
            result[i] = r
        return list(set(result))

    # pdf 이름 만들기
    def make_pdf_name(url):
        return url.split('/CustomsNotices/')[1].replace('/', '_')

    # url 주소에서 path로 pdf 저장하기
    def url_to_download(path, url):
        response = requests.get(url)
        dpath = path+make_pdf_name(url)
        with open(dpath, 'wb') as pdf:
            pdf.write(response.content)
        return dpath

    # page~ 이후의 값을 날려버림
    def get_summ(txt):
        if 'Page intentionally left blank' in txt:
            ret = txt.split('Page intentionally left blank')[0]
            return ret
        else:
            return txt

    # pdf에서 텍스트 긁어오기
    def get_contents(pdf_path):
        text = ''
        with open(pdf_path, 'rb') as pdf:
            for page_layout in extract_pages(pdf):
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        text += element.get_text().strip().replace('\n', '') + ' '
        return text
    
    def make_dates_au(txt):
        months = ['January','February','March','April','May','June','July','August','September','October','November','December']
        splited_txt = txt.split(' ')
        ret = ''
        for i,m in enumerate(months):
            if splited_txt[1] == m:
                ret += splited_txt[2] + '년 ' + str(i+1) + '월 ' + splited_txt[0] + "일 게시"
        return ret

    ######################################################
    ### 크롤링 파트
    ######################################################
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import time

    chrome_driver_path = '../baekho/chromedriver.exe'
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument("disable-blink-features=AutomationControlled") #selenium 엿먹이는 코드 다시 엿먹이기
    options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(chrome_driver_path, options=options)

    base_url = 'https://www.abf.gov.au'  #합격

    titles, dates, contents, files, links = [], [], [], [], []
    all_contents = []

    # driver.get(base_url+'/customs/302249/302266/index.html')
    driver.get(base_url+'/help-and-support/notices/australian-customs-notices')
    time.sleep(3)
    driver.maximize_window()
    WebDriverWait(driver, 200).until(EC.presence_of_element_located((By.XPATH, "/html/body/form/div[3]/div[2]/div[3]/div[3]/main/div/div[5]/div[2]/div/div[2]/div[3]/div/div/div/div[1]/ha-table-search/div/div[2]/div/div[2]/div/div")))

    html = driver.page_source
    soup = bs(html, 'html.parser')

    ts = soup.find_all('tr', 'accordion-header')
    des_cont = soup.find_all('tr', 'accordion-content')

    for i, t, d in zip(range(5), ts, des_cont):
        title = get_text(t.find_all('td')[1])
        titles.append(title[:-27])
        date = get_text(d.find_all('div', 'col-sm-3')[0].find('span'))
        dates.append(make_dates_au(date))
        file_url = base_url + d.find_all('div', 'col-sm-3')[1].find('a')['href']
        links.append(file_url)
        
        dpath = url_to_download('./', file_url)
        txt = get_contents(dpath) #pdf파일에서 내용 긁음
        txt = ' '.join(txt.split())
        
        contents.append(get_summ(txt))
        all_contents.append(txt)
    driver.close()

    ######################################################
    ### 번역 파트
    ######################################################
    print('translate...')

    translator = Translator()
    titles_tr = []
    contents_tr = ['-','-','-','-','-']
    for title in titles:
        title_tmp = translator.translate(title, src='en', dest="ko")
        title_translated = title_tmp.text
        titles_tr.append(title_translated)

        # cont_tmp = translator.translate(content ,src='en', dest="en")
        # cont_translated = rid_sc(cont_tmp.text)
        # contents_tr.append(cont_translated)
    
    ######################################################
    ### 키버트 파트
    ######################################################

    print('keyBERT...')

    keywords_list = [' ',' ',' ',' ',' ']

    kw_model = KeyBERT(model=model)
    for i, t in enumerate(contents):
        k = ''
        p = rid_sc(t)
        # use_mmr True 후 diversity로 나오는 친구들 다양성 조정
        # 마찬가지로 stop_words도 국가별로 바꿔주면 됨.
        keywords = kw_model.extract_keywords(p, keyphrase_ngram_range=(3,3), stop_words=au_sw, top_n=5, use_mmr=True,diversity=0.55)
        for j, keyword in enumerate(keywords):
            tmp = keyword[0].replace(' ', '')
            for jj in range(10-j*2):
                k += tmp + ' '
        cloud = WordCloud(prefer_horizontal=1,background_color='white',width=1200, height=200).generate(k)
        cloud.to_file('./baekho/static/warn/image/'+"호주-au"+str(i+1)+'.png')

    ##############################################################
    ## 키워드 hscode 비교 파트
    ##############################################################
    print('find goods...')

    ngrams, codes, names, sims = [],[],[],[]
    for c in contents:
        c = rid_sc(c)
        txt = generate_N_grams(c,2,origin_sw)
        ls = get_score(txt, data, 0.6)
        a,b,c,d = '','','',''
        for ng, co, na, si in zip(ls.ngram,ls.code,ls.name,ls.sim):
            if ng:
                a += ng + '||'
                if len(co) == 3:
                    co = '0' + co
                b += co + '||'        
                c += na + '||'        
                d += si + '||'  
            else:
                a,b,c,d = ' ',' ',' ',' '
        while True:
            if 'None||' in a:
                a = a.strip('None||')
                b = b.strip('None||')
                c = c.strip('None||')
                d = d.strip('an||')
            else:
                break
        a += ' '
        b += ' '
        c += ' '
        d += ' '
        
        ngrams.append(a)
        codes.append(b)    
        names.append(c)    
        sims.append(d)
    
    ##############################################################
    ##찐 hscode 검출 파트
    ##############################################################
    real_codes = []
    for v in all_contents:
        ls = get_hscode(v)
        if len(ls) == 0:
            ls.append('None')
        real_codes.append(ls)

    ##############################################################
    ## 요약본 만들기
    ##############################################################
    from transformers import pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summarizes, summarizes_tr = [], []
    print('summarizing...')
    for sum_content in contents:
        if len(sum_content) > 4000:
            sum_content = sum_content[:4000]
        summ = summarizer(sum_content, max_length=130, min_length=30, do_sample=False)
        summ_tr = translator.translate(summ[0]['summary_text'], src='en', dest="ko")
        summarizes.append(summ[0]['summary_text'])
        summarizes_tr.append(summ_tr.text)
    
    ##############################################################
    ##결과물 만들기 파트
    ##############################################################
    print('saving...')
    nums = range(1, len(titles)+1)
    rows = []
    for a,b,c,d,e, f,g,h,i,j, k,l,m,n,o in zip(nums, titles, titles_tr ,dates, contents, contents_tr, keywords_list, ngrams, codes, names, sims, real_codes, links, summarizes, summarizes_tr):
                                            #nums, titles, titles_tr, files, dates, contents, summarizes, summarizes_tr, keywords_list, ngrams, codes, names, sims, real_codes
        row= [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o]
        rows.append(row)

    result = pd.DataFrame(rows, columns=['번호','제목','제목(한국어번역)','날짜','내용','내용(영어번역)','키워드 요약','일치 키워드','hs코드','품목','유사도','본문에 진짜 코드','링크','요약본(영어)','요약본(한국어)'])
                                        #번호','제목','제목(한국어번역)','날짜','내용','내용(영어번역)','키워드 요약','일치 키워드','hs코드','품목','유사도','본문에 진짜 코드','링크'
    result.to_csv('./result_au.csv', index=False)

    dir = '../baekho'
    for f in os.listdir(dir):
        if f.lower().endswith('.pdf'):
            os.remove(os.path.join(dir, f))

    print('done')

    return 'result_au.csv'

def usa():
    return 
