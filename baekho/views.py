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
from baekho.models import Country, Opening, HeadOffice, Industry, Compare
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

from django.core.mail import EmailMessage
from django.template.loader import render_to_string
import smtplib
from email.utils import formataddr
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
def save_last():
    print("!!scheduler!! save_last 시작==")
    def save_last_title(CSV_PATH):
        file = open(CSV_PATH, encoding="utf8")
        reader = csv.reader(file)
        next(reader, None) 
        cnt = 0 
        for row in reader:
            if (cnt==0): 
                last_title = row[2]
            cnt += 1
        return last_title


    records = Compare.objects.all()
    records.delete()

    list = []
    list.append(Compare(
        last_china = save_last_title("result_cn.csv"),
        last_usa = save_last_title("result_us.csv"),
        last_japan = save_last_title("result_ja.csv"),
        last_vietnam = save_last_title("result_vi.csv"),
        last_australia = save_last_title("result_au.csv"),
    ))

    Compare.objects.bulk_create(list)

    print("last:",list)
    print("!!scheduler!! save_last 끝==")
    return 


def send_mail():
    def mail(country, CSV_PATH): 
        from_addr = formataddr(("백호", "mink1414@naver.com"))
        to_addr = formataddr(("코트라", "simsong88@naver.com"))

        session = None 

        # {중국} {2022-08-20} 고시에서 {Vessels for the transport of goods, Vessels for the transport of both persons and goods}({890190}) 이 검출되었습니다.
        # {url}

        # title, date, name, hscode, url 
        file = open(CSV_PATH, encoding="utf8")
        reader = csv.reader(file)
        next(reader, None)
        cnt = 0
        for row in reader: 
            cnt += 1
            if(cnt ==   1):
                title = row[2]
                date = row[3]
                name = row[9]
                hscode = row[8]
            break

        print("==info:",title, date, name, hscode )
        try:
            # SMTP 세션 생성
            session = smtplib.SMTP("smtp.naver.com", 587)
            session.set_debuglevel(True)

            # SMTP 계정 인증 설정
            session.ehlo()
            session.starttls()
            # session.login("mmm@naver.com", "mmmm") ### 이 부분 지금 오류나는데 실행할 때 이메일, 비밀번호 올바르게 넣어주면 됨
            session.login("mink1414@naver.com", "min20903")

            # 매일 콘텐츠 설정
            message = MIMEMultipart("alternative")

            # 메일 송/수신 옵션 설정
            message.set_charset("utf-8")
            message["From"] = from_addr
            message["To"] = to_addr
            message["Subject"] = "[데청캠 백호-KOTRA] 알람입니다."

            # 메일 콘텐츠 내용
            # {중국} {2022-08-20} 고시에서 {Vessels for the transport of goods, Vessels for the transport of both persons and goods}({890190}) 이 검출되었습니다.
           
            body = country +"-"+ title + " ("+date +") 고시에서 "+name+" ("+hscode+") 이 검출되었습니다." 
            # body = '''
            # <p>테스트용 메일입니다.. 무시하세요..</p>
            # '''

            bodyPart = MIMEText(body, "html", "utf-8")
            message.attach(bodyPart)

            # 메일 발송
            session.sendmail(from_addr, to_addr, message.as_string())

            print("Successfully sent the mail!!!")
        except Exception as e:
            print(e)

        finally:
            if session is not None:
                session.quit()
        return 

    # # 1) 새로운게 올라왔는지
    # # 2) 올라왔으면 hs코드 검출이 됐는지

    # def save_last_title(CSV_PATH):
    #     file = open(CSV_PATH, encoding="utf8")
    #     reader = csv.reader(file)
    #     next(reader, None) 
    #     cnt = 0 
    #     for row in reader:
    #         if (cnt==0): 
    #             last_title = row[2]
    #         cnt += 1
    #     return last_title

    # # 크롤링 스케줄러 돌리기 - 멀티쓰레드 x
    # china()
    # usa()
    # japan()
    # vietnam()
    # australia()

    # 함수 정의 - country, CSV_PATH
    def compare_last_now(country, country_eng, CSV_PATH):
        print("!!scheduler!! 메일 여부 확인==")
        file = open(CSV_PATH, encoding="utf8")
        isSend = False 
        reader = csv.reader(file)
        next(reader, None)
        cnt = 0
        for row in reader:
            if (cnt==0):
                now_title = row[2]
                ngram = row[7]
            cnt += 1
        print("!!scheduler now_title:",now_title)

        last_title = Compare.objects.all().values("last_"+country_eng)
        # last_title = save_last_title(CSV_PATH)
        if (last_title != now_title):
            print("last:",last_title) 
            print("now:", now_title)
            if (ngram != "['None']"):
                isSend=True
                mail(country, CSV_PATH)
                print("!!send_mail!! ==메일 보내기 성공==")
            else: 
                print("!!send_mail!! ==메일 안보내기(사유: hscode가 None임)==")
        else:
            print("!!send_mail!! ==메일 안보내기(사유: 새로올라온거없음)==")
        return 

    compare_last_now("중국","china", "result_cn.csv") 
    compare_last_now("미국","usa", "result_us.csv")
    compare_last_now("일본","japan", "result_ja.csv")
    compare_last_now("베트남","vietnam", "result_vi.csv")
    compare_last_now("호주","australia", "result_au.csv") 

    return 

def baekho_detail(request, pk):

    context = dict()
    country = Country.objects.get(id=pk)
    context["country"] = country

    countries = Country.objects.all()
    context["countries"] = countries
    now = datetime.datetime.now()
    print("======현재시간:",now,"=====")

    def make_context(CSV_PATH, name, name_eng):
            records = HeadOffice.objects.all()
            records.delete()

            file = open(CSV_PATH, encoding="utf8")
            reader = csv.reader(file)
            next(reader, None)
            list = []
            cnt = 0

            def str_to_list(txt):
                ret = []
                tmp = txt.replace('[', '').replace(']', '').replace(" '", '||').replace("'", '').split(',||')
                for t in tmp:
                    ret.append(t)

                while True:
                    if len(ret) < 5:
                        ret.append(' ')
                    else:
                        break
                return ret

            for row in reader:
                cnt += 1
                
                print(row[0]+"번째 출력\n")

                ngram_list = str_to_list(row[7])
                code_list = str_to_list(row[8])
                name_list = str_to_list(row[9])
                sim_list = str_to_list(row[10])

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
                                    img="/static/warn/image/"+str(name)+"-"+str(cnt)+".png",
                                    country=name,
                                    country_eng = name_eng,
                                    
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
                                    
                                    txt_eng = row[13],
                                    txt = row[14],
                                    korea = row[6],
                                    export = row[15],
                                    ))  

            HeadOffice.objects.bulk_create(list)

            opening = HeadOffice.objects.all()
            context["opening"] = opening

            return 
    
    if (pk==1): # 중국
        #CSV_PATH = china()
        CSV_PATH = "../baekho/result_cn.csv" 
        country = "중국"
        country_eng = "cn"

    elif (pk==2): # 미국
        # CSV_PATH = usa()
        CSV_PATH = "../baekho/result_us.csv" 
        country = "미국"
        country_eng = "us"
        
    elif (pk==3): # 일본
        
        # CSV_PATH = japan()
        CSV_PATH = "../baekho/result_ja.csv" 
        country = "일본"
        country_eng = "ja"

    elif (pk==4): # 베트남 
        #CSV_PATH = "../baekho/" + vietnam()
        CSV_PATH = "../baekho/result_vi.csv"   
        country = "베트남" 
        country_eng = "vi"

    elif (pk==5): # 호주 
        #CSV_PATH = "../baekho/" + australia()
        CSV_PATH = "../baekho/result_au.csv"
        country = "호주"
        country_eng = "au"
    
    make_context(CSV_PATH, country, country_eng)

    return render(request, 'warn/detail.html', context=context)

def subpage(request, pk, title):

    records = Industry.objects.all()
    records.delete()

    context = dict() 
    info = HeadOffice.objects.get(number=title)
    context["info"] = info

    country_eng = info.country_eng
    number = info.number
    file = open("find_"+country_eng+"_"+str(number)+".csv", encoding="utf8")
    reader = csv.reader(file)
    
    next(reader, None)
    list = []
    
    cnt = 0 
    for row in reader:
        cnt += 1
        print("row 2: ",row[2])
        print("row 3: ",row[3])

        if len(row[2]) == 5:
            row[2] = "0"+row[2]

        list.append(Industry(
            number = cnt,
            hscode = row[0],
            hscode_name = row[1],
            KSIC10 = row[2],
            KSIC10_name = row[3], 
        ))

    Industry.objects.bulk_create(list)
    
    industry = Industry.objects.all()
    context["industry"] = industry

    country = Country.objects.get(id=pk)
    context["country"] = country


    return render(request, "warn/subpage.html", context=context)

def vietnam():
    print("======베트남 시작=====")
    nltk.download('stopwords')
    origin_sw = stopwords.words('english')
    # cn_sw = origin_sw + ['china','chinese','customs', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    vt_sw = origin_sw + ['deputy', 'documents','minister','committee','committe','nguyen','vietnam','customs', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # data = pd.read_excel('../hs_vt_eng.xlsx')
    # data['hs_embedding'] = data.apply(lambda row: model.encode(row.subject), axis = 1) #hs품목
    with open('hs_embedded_vi.pkl', 'rb') as f: #개체의 특성을 피클모델이 이용할 수 있는 객체를 담아낸것
        data = pickle.load(f)
    with open('reg_embedded.pkl', 'rb') as f:
        reg = pickle.load(f)
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
            for a,b,c in zip(data['hs_embedding'], data['SUBJECT'], data['HSCODE']):
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
        ret = re.sub(r"[^a-zA-Z ]"," ",txt).lower()
        return ' '.join(ret.split())
            
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

    # 규제 정보 확인 함수
    def get_score_reg(ngrams, reg, thresh=0.75):
        rows=[]
        not_none = False
        for ngram in ngrams:
            r_old, r_new = 0, 0
            tmp = model.encode(ngram)
            for r, w in zip(reg['hs_embedding'], reg['WORDS']):
                row = []
                r_new = cos_sim(r, tmp)
                if r_old < r_new and r_new > thresh:
                    r_old = r_new
                    row.append(ngram)
                    row.append(w)
                    row.append(r_new)
                    rows.append(row)
        for ro in rows:
            if ro:
                not_none = True
        return not_none

    def make_ind_table(ind_hs_path=None, country_path=None, find_path=None,country=None): #입력값 없어도 실행하기 위해 None값을 줌
        i_path = ind_hs_path + 'ind_hs_code.csv'
        ind_hs_code = pd.read_csv(i_path, encoding='cp949')
        find_table = ind_hs_code.copy().iloc[:, [9,10,7,8]].astype(str) #관련산업군 데이터 추출
        
        for i, c in enumerate(find_table[' HS2017_CD']):
            if len(c) == 5: #0부터 시작하는 산업 코드 체크
                c = '0'+c
                find_table[' HS2017_CD'][i] = c

        def make_find_table(finded, find_table):
            ret = pd.DataFrame(columns=[' HS2017_CD', ' HS2017_KO_NM', ' KSIC10_CD', ' KSIC10_KO_NM'])
            for t in finded:
                for index, f in enumerate(find_table[' HS2017_CD']):
                    if t == f:
                        find = find_table.iloc[index, :]
                        ret = ret.append(find, ignore_index = True)
            ret = ret.drop_duplicates(keep = 'first')
            
            if len(ret) == 0:
                ret = pd.DataFrame([['None','None','None','None']],columns=[' HS2017_CD', ' HS2017_KO_NM', ' KSIC10_CD', ' KSIC10_KO_NM'])
            return ret

        c_path = country_path + 'result_' +country +'.csv'
        test_result = pd.read_csv(c_path).iloc[:, [11, 8]]
        #make_ind_table('','','','') #ind_hs코드 패쓰, reult_나라.csv파일 패쓰, find_나라.csv 다운받을 경로, 나라이름
        #make_ind_table('', '../result/', '../find/', 'au')
        a = test_result['본문에 진짜 코드']
        b = test_result['hs코드']

        def str_to_list(txt):
            ret = []
            tmp = txt.replace('[', '').replace(']', '').replace(" '", '').replace("'", '').split(',')
            for t in tmp:
                ret.append(t)
            return ret

        for i, t1, t2 in zip(range(1,6), a, b):
            if 'None' not in t1:
                ls = str_to_list(t1)
                path = find_path + 'find_' + country+'_' + str(i) + '.csv'
                make_find_table(ls, find_table).to_csv(path, index=False)
            else:
                ls = str_to_list(t2)
                path = find_path + 'find_' + country+'_' + str(i) + '.csv'
                make_find_table(ls, find_table).to_csv(path, index=False)
        
    #make_ind_table('', '', '', 'vi')     

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

    driver.get(base_url+'/index.jsp?pageId=4&cid=30')
    #time.sleep(5)
    driver.maximize_window()
    WebDriverWait(driver, 200).until(EC.presence_of_element_located((By.XPATH, "/html/body/div/div[2]/section[2]/div/div[1]/div[2]/div[1]/div[1]/div/module/div[1]")))
    html = driver.page_source
    soup = bs(html, 'html.parser')
    lists = soup.find('div', id='listItems').find_all('div','content_item')[:5]

    print("======베트남 크롤링 시작=====")  
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

    print("======베트남 번역 시작=====")  
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

    ######################################################
    ## 한국 검출
    ######################################################
    is_korea_list = []
    for c in contents_tr:
        is_korea = False
        if 'Korea' in c:
            is_korea = True
        is_korea_list.append(is_korea)

    ##############################################################
    ## 키버트 키워드 요약 파트
    ##############################################################
    print("======베트남 키워드 요약 시작=====")
    from keybert import KeyBERT

    dir = './baekho/static/warn/image'
    for f in os.listdir(dir):
        if f.lower().endswith('.png'):
            os.remove(os.path.join(dir, f))

    #keywords_list = [' ',' ',' ',' ',' ']
    kw_model = KeyBERT(model=model)
    for i, t in enumerate(contents_tr):
        k = ''
        p = rid_sc(t)
        # use_mmr True 후 diversity로 나오는 친구들 다양성 조정
        # 마찬가지로 stop_words도 국가별로 바꿔주면 됨.
        keywords = kw_model.extract_keywords(p, keyphrase_ngram_range=(3,3), stop_words=vt_sw, top_n=5, use_mmr=True,diversity=0.55)
        for j, keyword in enumerate(keywords):
            tmp = keyword[0].replace(' ', '_')
            for jj in range(10-j*2):
                k += tmp + ' '
        cloud = WordCloud(prefer_horizontal=1,background_color='white',width=1200, height=200).generate(k)

        # 저장할 때 국가 이름을 바꿔주면 된다. (ex, 호주-au, 중국-cn, 베트남-vi, 미국-us, 일본-ja)
        # 저장할 때 경로 알아서 바꾸셈
        cloud.to_file('./baekho/static/warn/image/'+"베트남-"+str(i+1)+'.png')

    ##############################################################
    ## 키워드 hscode 비교 파트
    ##############################################################

    ngrams, codes, names, sims, is_reg_list = [],[],[],[],[]
    for con in contents_tr:
        con = rid_sc(con)
        txt = generate_N_grams(con,2,origin_sw)
        ls = get_score(txt, data, 0.6)
        reg_txt = generate_N_grams(con, 5, origin_sw)
        is_reg_list.append(get_score_reg(reg_txt,reg))
        a,b,c,d = [],[],[],[]
        for ng, co, na, si in zip(ls.ngram,ls.code,ls.name,ls.sim):
            if ng:
                if len(co) == 5:
                    co = '0' + co
                a.append(ng)
                b.append(co)
                c.append(na)
                d.append(si)
            else:
                continue
        if len(a) == 0:
            a.append('None')
            b.append('None')
            c.append('None')
            d.append('None')
        ngrams.append(a)
        codes.append(b)    
        names.append(c)    
        sims.append(d)

    ##############################################################
    ##찐 hscode 검출 파트
    ##############################################################
    print("======베트남 hscode 검출 시작=====")
    real_codes = []
    for v in contents:
        ls = make_code(get_hscode(v))
        if len(ls) == 0:
            ls.append('None')
        real_codes.append(ls)

    ##############################################################
    ## 요약본 만들기
    ##############################################################
    from transformers import pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summarizes, summarizes_tr = [], []
    print('======베트남 summarizing=====')
    for sum_content in contents_tr: #베트남 내용 영어로 바꾼것
        if len(sum_content) > 4000:
            sum_content = sum_content[:4000]
        summ = summarizer(sum_content, max_length=130, min_length=30, do_sample=False)
        summ_tr = translator.translate(summ[0]['summary_text'], src='en', dest="ko")
        summarizes.append(summ[0]['summary_text']) #요약 영어
        summarizes_tr.append(summ_tr.text) #요약 한국어


    ##############################################################
    ##결과물 만들기 파트
    ##############################################################
    print("======베트남 csv 저장=====")

    nums = range(1, len(titles)+1)
    rows = []
    for a,b,c,d,e, f,g,h,i,j, k,l,m,n,o, p in zip(nums, titles, titles_tr, dates, contents, contents_tr, is_korea_list, ngrams, codes, names, sims, real_codes, links, summarizes, summarizes_tr, is_reg_list):
        row= [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p]
        rows.append(row)

    result = pd.DataFrame(rows, columns=['번호','제목','제목(한국어번역)','날짜','내용','내용(영어번역)','한국 검출','일치 키워드','hs코드','품목','유사도','본문에 진짜 코드','링크','요약본(영어)','요약본(한국어)', '규제정보'])
    result.to_csv('./result_vi.csv', index=False)

    path = "result_vi.csv"
    result.to_csv('./result_vi.csv', index=False)
    make_ind_table('', '', '', 'vi')
    print("======베트남 완료=====")
    return path
    
def china():
    ##############################################################
    ## 필요한 것들 미리 정의
    ##############################################################
    print("======중국 시작=====")
    nltk.download('stopwords')
    origin_sw = stopwords.words('english')
    cn_sw = origin_sw + ['china','chinese','customs', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    #vt_sw = origin_sw + ['deputy', 'documents','minister','committee','committe','nguyen','vietnam','customs', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    with open('hs_embedded_cn.pkl', 'rb') as f: #개체의 특성을 피클모델이 이용할 수 있는 객체를 담아낸것
        data = pickle.load(f)
    with open('reg_embedded.pkl', 'rb') as f:
        reg = pickle.load(f)
    # data1 = pd.read_excel('../hs_china.xlsx')
    # data = data1.loc[:,"number":"subject"]
    # data['hs_embedding'] = data.apply(lambda row: model.encode(row.subject), axis = 1) #hs품목 
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
            for a,b,c in zip(data['hs_embedding'], data['SUBJECT'], data['HSCODE']):
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
        return re.sub(r"[^a-zA-Z ]","",txt).lower()
        
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

    # 규제 정보 확인 함수
    def get_score_reg(ngrams, reg, thresh=0.75):
        rows=[]
        not_none = False
        for ngram in ngrams:
            r_old, r_new = 0, 0
            tmp = model.encode(ngram)
            for r, w in zip(reg['hs_embedding'], reg['WORDS']):
                row = []
                r_new = cos_sim(r, tmp)
                if r_old < r_new and r_new > thresh:
                    r_old = r_new
                    row.append(ngram)
                    row.append(w)
                    row.append(r_new)
                    rows.append(row)
        for ro in rows:
            if ro:
                not_none = True
        return not_none

    def make_ind_table(ind_hs_path=None, country_path=None, find_path=None,country=None): #입력값 없어도 실행하기 위해 None값을 줌
        i_path = ind_hs_path + 'ind_hs_code.csv'
        ind_hs_code = pd.read_csv(i_path, encoding='cp949')
        find_table = ind_hs_code.copy().iloc[:, [9,10,7,8]].astype(str) #관련산업군 데이터 추출
        
        for i, c in enumerate(find_table[' HS2017_CD']):
            if len(c) == 5: #0부터 시작하는 산업 코드 체크
                c = '0'+c
                find_table[' HS2017_CD'][i] = c

        def make_find_table(finded, find_table):
            ret = pd.DataFrame(columns=[' HS2017_CD', ' HS2017_KO_NM', ' KSIC10_CD', ' KSIC10_KO_NM'])
            for t in finded:
                for index, f in enumerate(find_table[' HS2017_CD']):
                    if t == f:
                        find = find_table.iloc[index, :]
                        ret = ret.append(find, ignore_index = True)
            ret = ret.drop_duplicates(keep = 'first')
            
            if len(ret) == 0:
                ret = pd.DataFrame([['None','None','None','None']],columns=[' HS2017_CD', ' HS2017_KO_NM', ' KSIC10_CD', ' KSIC10_KO_NM'])
            return ret

        c_path = country_path + 'result_' +country +'.csv'
        test_result = pd.read_csv(c_path).iloc[:, [11, 8]]
        #make_ind_table('','','','') #ind_hs코드 패쓰, reult_나라.csv파일 패쓰, find_나라.csv 다운받을 경로, 나라이름
        #make_ind_table('', '../result/', '../find/', 'au')
        a = test_result['본문에 진짜 코드']
        b = test_result['hs코드']

        def str_to_list(txt):
            ret = []
            tmp = txt.replace('[', '').replace(']', '').replace(" '", '').replace("'", '').split(',')
            for t in tmp:
                ret.append(t)
            return ret

        for i, t1, t2 in zip(range(1,6), a, b):
            if 'None' not in t1:
                ls = str_to_list(t1)
                path = find_path + 'find_' + country+'_' + str(i) + '.csv'
                make_find_table(ls, find_table).to_csv(path, index=False)
            else:
                ls = str_to_list(t2)
                path = find_path + 'find_' + country+'_' + str(i) + '.csv'
                make_find_table(ls, find_table).to_csv(path, index=False)

    # make_ind_table('', '', '', 'cn')  

    ##############################################################
    ## 크롤링 파트
    ##############################################################
    
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
    print("======중국 크롤링 시작=====")
    driver.get(base_url+'/eportal/ui?pageId=2480148&currentPage=1&moduleId=08654b53d1cc42478813b0b2feddcb57&staticRequest=yes')
    time.sleep(3)
    driver.maximize_window()
    WebDriverWait(driver, 200).until(EC.presence_of_element_located((By.XPATH, "/html/body/div[4]/div/div[2]/div/div[2]")))
    html = driver.page_source
    soup = bs(html, 'html.parser')
    lists = soup.find('ul', 'conList_ull').find_all('li')

    from tqdm import tqdm

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
    print("======중국 번역 시작=====")
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

    ######################################################
    ## 한국 검출 
    ######################################################
    is_korea_list = []
    for c in contents_tr:
        is_korea = False
        if 'Korea' in c:
            is_korea = True
        is_korea_list.append(is_korea)

    ##############################################################
    ## 키버트 키워드 요약 파트
    ##############################################################
    print("======중국 키워드 요약 시작=====")
    from keybert import KeyBERT

    #keywords_list = [' ',' ',' ',' ',' ']
    kw_model = KeyBERT(model=model)
    for i, t in enumerate(contents_tr):
        k = ''
        p = rid_sc(t)
        # use_mmr True 후 diversity로 나오는 친구들 다양성 조정
        # 마찬가지로 stop_words도 국가별로 바꿔주면 됨.
        keywords = kw_model.extract_keywords(p, keyphrase_ngram_range=(3,3), stop_words=cn_sw, top_n=5, use_mmr=True,diversity=0.55)
        for j, keyword in enumerate(keywords):
            tmp = keyword[0].replace(' ', '_')
            for jj in range(10-j*2):
                k += tmp + ' '
        cloud = WordCloud(prefer_horizontal=1,background_color='white',width=1200, height=200).generate(k)

        # 저장할 때 국가 이름을 바꿔주면 된다. (ex, 호주-au, 중국-cn, 베트남-vi, 미국-us, 일본-ja)
        # 저장할 때 경로 알아서 바꾸셈
        cloud.to_file('./baekho/static/warn/image/'+"중국-"+str(i+1)+'.png')

    ##############################################################
    ## 키워드 hscode 비교 파트
    ##############################################################

    ngrams, codes, names, sims, is_reg_list = [],[],[],[],[]
    for con in contents_tr:
        con = rid_sc(con)
        txt = generate_N_grams(con,2,origin_sw)
        ls = get_score(txt, data, 0.6)
        reg_txt = generate_N_grams(con, 5, origin_sw)
        is_reg_list.append(get_score_reg(reg_txt,reg))
        a,b,c,d = [],[],[],[]
        for ng, co, na, si in zip(ls.ngram,ls.code,ls.name,ls.sim):
            if ng:
                if len(co) == 5:
                    co = '0' + co
                a.append(ng)
                b.append(co)
                c.append(na)
                d.append(si)
            else:
                continue
        if len(a) == 0:
            a.append('None')
            b.append('None')
            c.append('None')
            d.append('None')
        ngrams.append(a)
        codes.append(b)    
        names.append(c)    
        sims.append(d)

    ##############################################################
    ##찐 hscode 검출 파트
    ##############################################################
    print("======중국 hscode 검출 시작=====")
    real_codes = []
    for v in contents:
        ls = make_code(get_hscode(v))
        if len(ls) == 0:
            ls.append('None')
        real_codes.append(ls)

    ##############################################################
    ## 요약본 만들기
    ##############################################################
    from transformers import pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summarizes, summarizes_tr = [], []
    print('======중국 summarizing=====')
    for sum_content in contents_tr: #베트남 내용 영어로 바꾼것
        if len(sum_content) > 4000:
            sum_content = sum_content[:4000]
        summ = summarizer(sum_content, max_length=130, min_length=30, do_sample=False)
        summ_tr = translator.translate(summ[0]['summary_text'], src='en', dest="ko")
        summarizes.append(summ[0]['summary_text']) #요약 영어
        summarizes_tr.append(summ_tr.text) #요약 한국어

    ##############################################################
    ##결과물 만들기 파트
    ##############################################################
    print("======중국 csv 저장=====")
    nums = range(1, len(titles)+1)
    rows = []
    for a,b,c,d,e, f,g,h,i,j, k,l,m,n,o, p in zip(nums, titles, titles_tr, dates, contents, contents_tr, is_korea_list, ngrams, codes, names, sims, real_codes, links, summarizes, summarizes_tr, is_reg_list):
        row= [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p]
        rows.append(row)

    result = pd.DataFrame(rows, columns=['번호','제목','제목(한국어번역)','날짜','내용','내용(영어번역)','한국 검출','일치 키워드','hs코드','품목','유사도','본문에 진짜 코드','링크','요약본(영어)','요약본(한국어)', '규제정보'])
    result.to_csv('./result_cn.csv', index=False)
    make_ind_table('', '', '', 'cn') 
    print("======중국 완료=====")
    return "result_cn.csv"

def japan():
    print("======일본 시작=====")
    warnings.filterwarnings('ignore')

    ######################################################
    ### 필요한 것들 정의
    ######################################################
    nltk.download('stopwords')
    origin_sw = stopwords.words('english')
    # cn_sw = origin_sw + ['china','chinese','customs', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    # vt_sw = origin_sw + ['deputy', 'documents','minister','committee','committe','nguyen','vietnam','customs', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    # au_sw = origin_sw + ['australian','january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # data = pd.read_excel('../hscode/hscode.xlsx')
    # data['hs_embedding'] = data.apply(lambda row: model.encode(row.SUBJECT), axis = 1)

    # data의 pkl화 된 데이터프레임 미리 로딩
    # 아래 path를 바꿔주면 거기서 로드할수 있음
    with open('hs_embedded_ja.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('reg_embedded.pkl', 'rb') as f:
        reg = pickle.load(f)

    ######################################################
    ### 함수 정의
    ######################################################

    # 코사인 유사도 계산
    def cos_sim(A, B):
        return dot(A, B)/(norm(A)*norm(B))

    # beautiful_soup 텍스트 추출
    def get_text(soup):
        return soup.get_text().replace('\n', ' ').strip()

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
        
        for i, r in enumerate(result):
            r = r.replace('.','')
            result[i] = r
        
        return list(set(result))

    # 긁어온 날짜 텍스트 한국어로 변환 (일본)
    def make_dates_ja(txt):
        splited_txt = txt.replace('年','년 ').replace('月','월 ').replace('日', '일').split(' ')
        ret = splited_txt[0] + ' ' + splited_txt[1] + ' ' + splited_txt[2] + " 게시"
        return ret
    
    # 규제 정보 확인 함수
    def get_score_reg(ngrams, reg, thresh=0.75):
        rows=[]
        not_none = False
        for ngram in ngrams:
            r_old, r_new = 0, 0
            tmp = model.encode(ngram)
            for r, w in zip(reg['hs_embedding'], reg['WORDS']):
                row = []
                r_new = cos_sim(r, tmp)
                if r_old < r_new and r_new > thresh:
                    r_old = r_new
                    row.append(ngram)
                    row.append(w)
                    row.append(r_new)
                    rows.append(row)
        for ro in rows:
            if ro:
                not_none = True
        return not_none
    
    def make_ind_table(ind_hs_path=None, country_path=None, find_path=None,country=None): #입력값 없어도 실행하기 위해 None값을 줌
        i_path = ind_hs_path + 'ind_hs_code.csv'
        ind_hs_code = pd.read_csv(i_path, encoding='cp949')
        find_table = ind_hs_code.copy().iloc[:, [9,10,7,8]].astype(str) #관련산업군 데이터 추출
        
        for i, c in enumerate(find_table[' HS2017_CD']):
            if len(c) == 5: #0부터 시작하는 산업 코드 체크
                c = '0'+c
                find_table[' HS2017_CD'][i] = c

        def make_find_table(finded, find_table):
            ret = pd.DataFrame(columns=[' HS2017_CD', ' HS2017_KO_NM', ' KSIC10_CD', ' KSIC10_KO_NM'])
            for t in finded:
                for index, f in enumerate(find_table[' HS2017_CD']):
                    if t == f:
                        find = find_table.iloc[index, :]
                        ret = ret.append(find, ignore_index = True)
            ret = ret.drop_duplicates(keep = 'first')
            
            if len(ret) == 0:
                ret = pd.DataFrame([['None','None','None','None']],columns=[' HS2017_CD', ' HS2017_KO_NM', ' KSIC10_CD', ' KSIC10_KO_NM'])
            return ret

        c_path = country_path + 'result_' +country +'.csv'
        test_result = pd.read_csv(c_path).iloc[:, [11, 8]]
        #make_ind_table('','','','') #ind_hs코드 패쓰, reult_나라.csv파일 패쓰, find_나라.csv 다운받을 경로, 나라이름
        #make_ind_table('', '../result/', '../find/', 'au')
        a = test_result['본문에 진짜 코드']
        b = test_result['hs코드']

        def str_to_list(txt):
            ret = []
            tmp = txt.replace('[', '').replace(']', '').replace(" '", '').replace("'", '').split(',')
            for t in tmp:
                ret.append(t)
            return ret

        for i, t1, t2 in zip(range(1,6), a, b):
            if 'None' not in t1:
                ls = str_to_list(t1)
                path = find_path + 'find_' + country+'_' + str(i) + '.csv'
                make_find_table(ls, find_table).to_csv(path, index=False)
            else:
                ls = str_to_list(t2)
                path = find_path + 'find_' + country+'_' + str(i) + '.csv'
                make_find_table(ls, find_table).to_csv(path, index=False)
    
    ######################################################
    ### 크롤링 파트
    ######################################################
    chrome_driver_path = '../baekho/chromedriver.exe'
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument("disable-blink-features=AutomationControlled") #selenium 엿먹이는 코드 다시 엿먹이기
    options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(chrome_driver_path, options=options)

    base_url = 'https://www.meti.go.jp'  #합격

    titles, dates, contents, links = [], [], [], []

    print("======일본 크롤링 시작=====")

    driver.get(base_url+'/press/category_02.html')
    time.sleep(3)
    driver.maximize_window()

    html = driver.page_source
    soup = bs(html, 'html.parser')

    title_soup = soup.find('div', 'main w1000').find_all('a')
    date_soup = soup.find('div', 'main w1000').find_all('dt')

    for i, d, t in zip(range(5), date_soup, title_soup):
        date = get_text(d)
        dates.append(make_dates_ja(date))
        titles.append(get_text(t))
        cont_url = base_url + t['href']
        links.append(cont_url)
        driver.get(cont_url)
        time.sleep(2)
        page_html = driver.page_source
        page_soup = bs(page_html, 'html.parser')
        
        cont = page_soup.find('div', 'main w1000')
        contents.append(get_text(cont))

    driver.close()
    ##############################################################
    ### 번역 파트
    ##############################################################
    print("======일본 번역 시작=====")
    translator = Translator()
    titles_tr, contents_tr= [],[]

    for content, title in zip(contents, titles):
        if len(content) > 4000:
            c1 = content[:4000]
            c2 = content[4001:]
            c1_tmp = translator.translate(c1,src='ja', dest="en" )
            c2_tmp = translator.translate(c2,src='ja', dest="en" )
            cont_tmp = c1_tmp.text + c2_tmp.text
            cont_translated = cont_tmp    
        else:
            cont_tmp = translator.translate(content ,src='ja', dest="en")
            cont_translated = cont_tmp.text
        title_tmp = translator.translate(title, src='ja', dest="ko")
        title_translated = title_tmp.text
        contents_tr.append(cont_translated)
        titles_tr.append(title_translated)

    ######################################################
    ## 한국 검출
    ######################################################
    is_korea_list = []
    for c in contents_tr:
        is_korea = False
        if 'Korea' in c:
            is_korea = True
        is_korea_list.append(is_korea)

    ######################################################
    ### 키버트 파트
    ######################################################
    print("======일본 키워드 요약 시작=====")

    from keybert import KeyBERT

    kw_model = KeyBERT(model=model)
    for i, t in enumerate(contents_tr):
        k = ''
        p = rid_sc(t)
        # use_mmr True 후 diversity로 나오는 친구들 다양성 조정
        # 마찬가지로 stop_words도 국가별로 바꿔주면 됨.
        keywords = kw_model.extract_keywords(p, keyphrase_ngram_range=(3,3), stop_words=origin_sw, top_n=5, use_mmr=True,diversity=0.55)
        for j, keyword in enumerate(keywords):
            tmp = keyword[0].replace(' ', '_')
            for jj in range(10-j*2):
                k += tmp + ' '
        cloud = WordCloud(prefer_horizontal=1,background_color='white',width=1200, height=200).generate(k)
        
        # 저장할 때 국가 이름을 바꿔주면 된다. (ex, 호주-au, 중국-cn, 베트남-vi, 미국-us, 일본-jp)
        # 저장할 때 경로 알아서 바꾸셈
        cloud.to_file('./baekho/static/warn/image/'+"일본-"+str(i+1)+'.png')

    ##############################################################
    ## 키워드 hscode 비교 파트
    ##############################################################

    ngrams, codes, names, sims, is_reg_list = [],[],[],[],[]
    for con in contents_tr:
        con = rid_sc(con)
        txt = generate_N_grams(con,2,origin_sw)
        ls = get_score(txt, data, 0.6)
        reg_txt = generate_N_grams(con, 5, origin_sw)
        is_reg_list.append(get_score_reg(reg_txt,reg))
        a,b,c,d = [],[],[],[]
        for ng, co, na, si in zip(ls.ngram,ls.code,ls.name,ls.sim):
            if ng:
                if len(co) == 5:
                    co = '0' + co
                a.append(ng)
                b.append(co)
                c.append(na)
                d.append(si)
            else:
                continue
        if len(a) == 0:
            a.append('None')
            b.append('None')
            c.append('None')
            d.append('None')
        ngrams.append(a)
        codes.append(b)    
        names.append(c)    
        sims.append(d)

    ##############################################################
    ##찐 hscode 검출 파트
    ##############################################################
    print("======일본 hscode 검출 시작=====")
    real_codes = []
    for v in contents_tr:
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


    print('======일본 summarizing=====')


    for sum_content in contents_tr:
        if len(sum_content) > 4000:
            sum_content = sum_content[:4000]
        summ = summarizer(sum_content, max_length=130, min_length=30, do_sample=False)
        tmp = summ[0]['summary_text'] + " "
        summ_tr = translator.translate(tmp, src='en', dest="ko")
        summarizes.append(tmp)
        summarizes_tr.append(summ_tr.text)
        

    ##############################################################
    ##결과물 만들기 파트
    ##############################################################

    print("======일본 csv 저장=====")

    nums = range(1, len(titles)+1)
    rows = []
    for a,b,c,d,e, f,g,h,i,j, k,l,m,n,o, p in zip(nums, titles, titles_tr, dates, contents, contents_tr, is_korea_list, ngrams, codes, names, sims, real_codes, links, summarizes, summarizes_tr, is_reg_list):
        row= [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p]
        rows.append(row)

    result = pd.DataFrame(rows, columns=['번호','제목','제목(한국어번역)','날짜','내용','내용(영어번역)','한국 검출','일치 키워드','hs코드','품목','유사도','본문에 진짜 코드','링크','요약본(영어)','요약본(한국어)', '규제정보'])
    result.to_csv('./result_ja.csv', index=False)
    make_ind_table('', '', '', 'ja')
    print("======일본 완료=====")
    return 'result_ja.csv'

def australia():
    print("======호주 시작=====")
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
    with open('hs_embedded_au.pkl', 'rb') as f: #개체의 특성을 피클모델이 이용할 수 있는 객체를 담아낸것
        data = pickle.load(f)
    with open('reg_embedded.pkl', 'rb') as f:
        reg = pickle.load(f)
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
    
    # 규제 정보 확인 함수
    def get_score_reg(ngrams, reg, thresh=0.75):
        rows=[]
        not_none = False
        for ngram in ngrams:
            r_old, r_new = 0, 0
            tmp = model.encode(ngram)
            for r, w in zip(reg['hs_embedding'], reg['WORDS']):
                row = []
                r_new = cos_sim(r, tmp)
                if r_old < r_new and r_new > thresh:
                    r_old = r_new
                    row.append(ngram)
                    row.append(w)
                    row.append(r_new)
                    rows.append(row)
        for ro in rows:
            if ro:
                not_none = True
        return not_none

    def make_ind_table(ind_hs_path=None, country_path=None, find_path=None,country=None): #입력값 없어도 실행하기 위해 None값을 줌
        i_path = ind_hs_path + 'ind_hs_code.csv'
        ind_hs_code = pd.read_csv(i_path, encoding='cp949')
        find_table = ind_hs_code.copy().iloc[:, [9,10,7,8]].astype(str) #관련산업군 데이터 추출
        
        for i, c in enumerate(find_table[' HS2017_CD']):
            if len(c) == 5: #0부터 시작하는 산업 코드 체크
                c = '0'+c
                find_table[' HS2017_CD'][i] = c

        def make_find_table(finded, find_table):
            ret = pd.DataFrame(columns=[' HS2017_CD', ' HS2017_KO_NM', ' KSIC10_CD', ' KSIC10_KO_NM'])
            for t in finded:
                for index, f in enumerate(find_table[' HS2017_CD']):
                    if t == f:
                        find = find_table.iloc[index, :]
                        ret = ret.append(find, ignore_index = True)
            ret = ret.drop_duplicates(keep = 'first')
            
            if len(ret) == 0:
                ret = pd.DataFrame([['None','None','None','None']],columns=[' HS2017_CD', ' HS2017_KO_NM', ' KSIC10_CD', ' KSIC10_KO_NM'])
            return ret

        c_path = country_path + 'result_' +country +'.csv'
        test_result = pd.read_csv(c_path).iloc[:, [11, 8]]
        #make_ind_table('','','','') #ind_hs코드 패쓰, reult_나라.csv파일 패쓰, find_나라.csv 다운받을 경로, 나라이름
        #make_ind_table('', '../result/', '../find/', 'au')
        a = test_result['본문에 진짜 코드']
        b = test_result['hs코드']

        def str_to_list(txt):
            ret = []
            tmp = txt.replace('[', '').replace(']', '').replace(" '", '').replace("'", '').split(',')
            for t in tmp:
                ret.append(t)
            return ret

        for i, t1, t2 in zip(range(1,6), a, b):
            if 'None' not in t1:
                ls = str_to_list(t1)
                path = find_path + 'find_' + country+'_' + str(i) + '.csv'
                make_find_table(ls, find_table).to_csv(path, index=False)
            else:
                ls = str_to_list(t2)
                path = find_path + 'find_' + country+'_' + str(i) + '.csv'
                make_find_table(ls, find_table).to_csv(path, index=False)
        
    #make_ind_table('', '', '', 'au')

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

    print("======호주 크롤링 시작=====")

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
    print("======호주 번역 시작=====")

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
    ## 한국 검출 
    ######################################################
    is_korea_list = []
    for c in contents:
        is_korea = False
        if 'Korea' in c:
            is_korea = True
        is_korea_list.append(is_korea)
    
    ######################################################
    ### 키버트 파트
    ######################################################

    print("======호주 키워드 요약 시작=====")
    #keywords_list = [' ',' ',' ',' ',' ']

    kw_model = KeyBERT(model=model)
    for i, t in enumerate(contents):
        k = ''
        p = rid_sc(t)
        # use_mmr True 후 diversity로 나오는 친구들 다양성 조정
        # 마찬가지로 stop_words도 국가별로 바꿔주면 됨.
        keywords = kw_model.extract_keywords(p, keyphrase_ngram_range=(3,3), stop_words=au_sw, top_n=5, use_mmr=True,diversity=0.55)
        for j, keyword in enumerate(keywords):
            tmp = keyword[0].replace(' ', '_')
            for jj in range(10-j*2):
                k += tmp + ' '
        cloud = WordCloud(prefer_horizontal=1,background_color='white',width=1200, height=200).generate(k)
        cloud.to_file('./baekho/static/warn/image/'+"호주-"+str(i+1)+'.png')
    ##############################################################
    ## 키워드 hscode 비교 파트
    ##############################################################
    print("======호주 hscode 검출 시작=====")


    ngrams, codes, names, sims, is_reg_list = [],[],[],[],[]
    for con in contents_tr:
        con = rid_sc(con)
        txt = generate_N_grams(con,2,origin_sw)
        ls = get_score(txt, data, 0.6)
        reg_txt = generate_N_grams(con, 5, origin_sw)
        is_reg_list.append(get_score_reg(reg_txt,reg))
        a,b,c,d = [],[],[],[]
        for ng, co, na, si in zip(ls.ngram,ls.code,ls.name,ls.sim):
            if ng:
                if len(co) == 5:
                    co = '0' + co
                a.append(ng)
                b.append(co)
                c.append(na)
                d.append(si)
            else:
                continue
        if len(a) == 0:
            a.append('None')
            b.append('None')
            c.append('None')
            d.append('None')
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
    print('======호주 summarizing=====')
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
    print("======호주 csv 저장=====")
    nums = range(1, len(titles)+1)
    rows = []
    for a,b,c,d,e, f,g,h,i,j, k,l,m,n,o, p in zip(nums, titles, titles_tr, dates, contents, contents_tr, is_korea_list, ngrams, codes, names, sims, real_codes, links, summarizes, summarizes_tr, is_reg_list):
        row= [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p]
        rows.append(row)

    result = pd.DataFrame(rows, columns=['번호','제목','제목(한국어번역)','날짜','내용','내용(영어번역)','한국 검출','일치 키워드','hs코드','품목','유사도','본문에 진짜 코드','링크','요약본(영어)','요약본(한국어)', '규제정보'])
    result.to_csv('./result_au.csv', index=False)
    make_ind_table('', '', '', 'au')
    dir = '../baekho'
    for f in os.listdir(dir):
        if f.lower().endswith('.pdf'):
            os.remove(os.path.join(dir, f))
    print("======호주 완료=====")
    return 'result_au.csv'

def usa():
    print("======미국 시작=====")
    warnings.filterwarnings('ignore')

    ######################################################
    ### 필요한 것들 정의
    ######################################################
    nltk.download('stopwords')
    origin_sw = stopwords.words('english')
    # cn_sw = origin_sw + ['china','chinese','customs', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    # vt_sw = origin_sw + ['deputy', 'documents','minister','committee','committe','nguyen','vietnam','customs', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    au_sw = origin_sw + ['australian','january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    us_sw = origin_sw + ['cbp','border','protection','january', 'february', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december', 'must']
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # data = pd.read_excel('../hscode/hscode.xlsx')
    # data['hs_embedding'] = data.apply(lambda row: model.encode(row.SUBJECT), axis = 1)

    # data의 pkl화 된 데이터프레임 미리 로딩
    # 아래 path를 바꿔주면 거기서 로드할수 있음
    with open('hs_embedded_us.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('reg_embedded.pkl', 'rb') as f:
        reg = pickle.load(f)

    ######################################################
    ### 함수 정의
    ######################################################

    # 코사인 유사도 계산
    def cos_sim(A, B):
        return dot(A, B)/(norm(A)*norm(B))

    # beautiful_soup 텍스트 추출
    def get_text(soup):
        return soup.get_text().replace('\n', ' ').replace('\u3000','').strip()

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
        
        for i, r in enumerate(result):
            r = r.replace('.','')
            result[i] = r
        
        return list(set(result))

    # pdf 이름 만들기
    def make_pdf_name(url):
        return url.split('/documents/')[1].replace('/', '_')

    # url 주소에서 path로 pdf 저장하기
    def url_to_download(path, url):
        response = requests.get(url)
        dpath = path+make_pdf_name(url)
        with open(dpath, 'wb') as pdf:
            pdf.write(response.content)
        return dpath

    # pdf에서 텍스트 긁어오기
    def get_contents(pdf_path):
        text, summary = '', ''
        with open(pdf_path, 'rb') as pdf:
            for page_layout in extract_pages(pdf):
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        k = element.get_text().strip().replace('-\n', '').replace('\n', ' ').replace('ﬁ', 'fi').replace('ﬂ','fl')
                        try: 
                            int(k)
                        except:
                            if not k.startswith('CUSTOMS BULLETIN AND DECISIONS'):
                                k = ' '.join(k.split())
                                text += k + ' '
                                if k.startswith('SUMMARY:'):
                                    k = ' '.join(k.split())
                                    summary += k + '\n'
        if 'U.S. Court of' in text:
            text = text.split('U.S. Court of')[0]
        return text.replace('- ',''), summary.replace('- ','')
        

    # SUMMARY 부분만 가져오기
    def get_summ(txt):
        ret = ''
        summed_ls = txt.split('SUMMARY')
        i = 1
        for s in summed_ls:
            if s.startswith(':'):
                ret += "NOTICE " + str(i) + s + '\n'
                i += 1
        return ret

    # 규제 정보 확인 함수
    def get_score_reg(ngrams, reg, thresh=0.75):
        rows=[]
        not_none = False
        for ngram in ngrams:
            r_old, r_new = 0, 0
            tmp = model.encode(ngram)
            for r, w in zip(reg['hs_embedding'], reg['WORDS']):
                row = []
                r_new = cos_sim(r, tmp)
                if r_old < r_new and r_new > thresh:
                    r_old = r_new
                    row.append(ngram)
                    row.append(w)
                    row.append(r_new)
                    rows.append(row)
        for ro in rows:
            if ro:
                not_none = True
        return not_none

    # 긁어온 날짜 텍스트 한국어로 변환 (미국)
    def make_dates_us(txt):
        splited_txt = txt.split(' ')[1].split('/')
        if splited_txt[0].startswith('0'):
            splited_txt[0] = splited_txt[0][1:]
        if splited_txt[1].startswith('0'):
            splited_txt[1] = splited_txt[1][1:]
        ret = splited_txt[2] + '년 ' + splited_txt[0] + '월 ' + splited_txt[1] + "일 게시"
        return ret

    # 없을 경우 이걸루
    def make_dates_us_mod(txt):
        months = ['January','February','March','April','May','June','July','August','September','October','November','December']
        splited_txt = txt.replace(',','').split(' ')
        ret = ''
        for i,m in enumerate(months):
            if splited_txt[0] == m:
                ret += splited_txt[2] + '년 ' + str(i+1) + '월 ' + splited_txt[1] + "일 게시"
        return ret

    def make_ind_table(ind_hs_path=None, country_path=None, find_path=None,country=None): #입력값 없어도 실행하기 위해 None값을 줌
        i_path = ind_hs_path + 'ind_hs_code.csv'
        ind_hs_code = pd.read_csv(i_path, encoding='cp949')
        find_table = ind_hs_code.copy().iloc[:, [9,10,7,8]].astype(str) #관련산업군 데이터 추출
        
        for i, c in enumerate(find_table[' HS2017_CD']):
            if len(c) == 5: #0부터 시작하는 산업 코드 체크
                c = '0'+c
                find_table[' HS2017_CD'][i] = c

        def make_find_table(finded, find_table):
            ret = pd.DataFrame(columns=[' HS2017_CD', ' HS2017_KO_NM', ' KSIC10_CD', ' KSIC10_KO_NM'])
            for t in finded:
                for index, f in enumerate(find_table[' HS2017_CD']):
                    if t == f:
                        find = find_table.iloc[index, :]
                        ret = ret.append(find, ignore_index = True)
            ret = ret.drop_duplicates(keep = 'first')
            
            if len(ret) == 0:
                ret = pd.DataFrame([['None','None','None','None']],columns=[' HS2017_CD', ' HS2017_KO_NM', ' KSIC10_CD', ' KSIC10_KO_NM'])
            return ret

        c_path = country_path + 'result_' +country +'.csv'
        test_result = pd.read_csv(c_path).iloc[:, [11, 8]]
        #make_ind_table('','','','') #ind_hs코드 패쓰, reult_나라.csv파일 패쓰, find_나라.csv 다운받을 경로, 나라이름
        #make_ind_table('', '../result/', '../find/', 'au')
        a = test_result['본문에 진짜 코드']
        b = test_result['hs코드']

        def str_to_list(txt):
            ret = []
            tmp = txt.replace('[', '').replace(']', '').replace(" '", '').replace("'", '').split(',')
            for t in tmp:
                ret.append(t)
            return ret

        for i, t1, t2 in zip(range(1,6), a, b):
            if 'None' not in t1:
                ls = str_to_list(t1)
                path = find_path + 'find_' + country+'_' + str(i) + '.csv'
                make_find_table(ls, find_table).to_csv(path, index=False)
            else:
                ls = str_to_list(t2)
                path = find_path + 'find_' + country+'_' + str(i) + '.csv'
                make_find_table(ls, find_table).to_csv(path, index=False)

    ######################################################
    ### 크롤링 파트
    ######################################################
    chrome_driver_path = '../baekho/chromedriver.exe'
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument("disable-blink-features=AutomationControlled") #selenium 엿먹이는 코드 다시 엿먹이기
    options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(chrome_driver_path, options=options)

    base_url = 'https://www.cbp.gov'  #합격

    titles, dates, contents, links = [], [], [], []
    all_contents = []
    print("======미국 크롤링 시작=====")
    # driver.get(base_url+'/customs/302249/302266/index.html')
    driver.get(base_url+'/trade/rulings/bulletin-decisions')
    time.sleep(3)

    html = driver.page_source
    soup = bs(html, 'html.parser')

    ls = soup.find('div', 'usa-accordion margin-bottom-3').find('div','usa-accordion__content').find_all('ul')

    for li in ls[:5]:
        page_url = base_url + li.find('a')['href']
        
        driver.get(page_url)
        time.sleep(3)
        page_html = driver.page_source
        page_soup = bs(page_html,'html.parser')
        
        titles.append(get_text(page_soup.find('h1','uswds-page-title page-title')))
        try:
            date = get_text(page_soup.find('time', 'datetime'))
            dates.append(make_dates_us(date))
        except:
            date = get_text(page_soup.find('span', 'field-content'))
            dates.append(make_dates_us_mod(date))
            
        file_url = base_url + page_soup.find('div', 'field-content').find('a')['href']
        links.append(file_url)
        
        dpath = url_to_download('./', file_url)
        txt, summary = get_contents(dpath)
        
        contents.append(get_summ(summary))
        all_contents.append(txt)
        
    driver.close()
    
    ######################################################
    ### 번역 파트
    ######################################################
    print("======미국 번역 시작=====")

    translator = Translator()
    titles_tr, contents_tr= [],['-','-','-','-','-']

    for title in titles:
        title_tmp = translator.translate(title, src='en', dest="ko")
        title_translated = title_tmp.text
        titles_tr.append(title_translated)
       
    ######################################################
    ### 키버트 파트
    ######################################################
    print("======미국 키워드 요약 시작=====")

    from wordcloud import WordCloud
    from keybert import KeyBERT

    kw_model = KeyBERT(model=model)
    for i, t in enumerate(contents):
        k = ''
        p = rid_sc(t)
        # use_mmr True 후 diversity로 나오는 친구들 다양성 조정
        # 마찬가지로 stop_words도 국가별로 바꿔주면 됨.
        keywords = kw_model.extract_keywords(p, keyphrase_ngram_range=(3,3), stop_words=us_sw, top_n=5, use_mmr=True,diversity=0.55)
        for j, keyword in enumerate(keywords):
            tmp = keyword[0].replace(' ', '_')
            for jj in range(10-j*2):
                k += tmp + ' '
        cloud = WordCloud(prefer_horizontal=1,background_color='white',width=1200, height=200).generate(k)
        
        # 저장할 때 국가 이름을 바꿔주면 된다. (ex, 호주-au, 중국-cn, 베트남-vi, 미국-us, 일본-jp)
        # 저장할 때 경로 알아서 바꾸셈
        cloud.to_file('./baekho/static/warn/image/'+"미국-"+str(i+1)+'.png')
        
    ##############################################################
    ## 키워드 hscode 비교 파트 & 규제 정보 확인
    ##############################################################
    print("======미국 hscode 검출 시작=====")

    ngrams, codes, names, sims, is_reg_list = [],[],[],[],[]
    for con in contents_tr:
        con = rid_sc(con)
        txt = generate_N_grams(con,2,origin_sw)
        ls = get_score(txt, data, 0.6)
        reg_txt = generate_N_grams(con, 5, origin_sw)
        is_reg_list.append(get_score_reg(reg_txt,reg))
        a,b,c,d = [],[],[],[]
        for ng, co, na, si in zip(ls.ngram,ls.code,ls.name,ls.sim):
            if ng:
                if len(co) == 5:
                    co = '0' + co
                a.append(ng)
                b.append(co)
                c.append(na)
                d.append(si)
            else:
                continue
        if len(a) == 0:
            a.append('None')
            b.append('None')
            c.append('None')
            d.append('None')
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
    summarizes ,summarizes_tr = [],[]
    # for summ in contents:
    #     summ_tr = translator.translate(summ, src='en', dest='ko')
    #     summarizes_tr.append(summ_tr.text)
    print('======미국 summarizing=====')


    for sum_content in contents:
        split_sum_content = sum_content.split('\n')
        tmp = ''
        i = 1
        for s in split_sum_content:
            if len(s) > 0:
                summ = summarizer(s, max_length=130, min_length=30, do_sample=False)
                tmp += 'NOTICE ' + str(i) + ': ' + summ[0]['summary_text'] + ' '  
                i+=1

        summ_tr = translator.translate(tmp, src='en', dest="ko")
        summarizes.append(tmp)
        summarizes_tr.append(summ_tr.text)
        

    ######################################################
    ## 한국 검출
    ######################################################
    is_korea_list = []
    for c in all_contents:
        is_korea = False
        if 'Korea' in c:
            is_korea = True
        is_korea_list.append(is_korea)
        
    ##############################################################
    ##결과물 만들기 파트
    ##############################################################

    print("======미국 csv 저장=====")

    nums = range(1, len(titles)+1)
    rows = []
    for a,b,c,d,e, f,g,h,i,j, k,l,m,n,o, p in zip(nums, titles, titles_tr, dates, contents, contents_tr, is_korea_list, ngrams, codes, names, sims, real_codes, links, summarizes, summarizes_tr, is_reg_list):
        row= [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p]
        rows.append(row)

    result = pd.DataFrame(rows, columns=['번호','제목','제목(한국어번역)','날짜','내용','내용(영어번역)','한국 검출','일치 키워드','hs코드','품목','유사도','본문에 진짜 코드','링크','요약본(영어)','요약본(한국어)', '규제정보'])

    result.to_csv('./result_us.csv', index=False)
    make_ind_table('', '', '', 'us')
    dir = '../baekho'
    for f in os.listdir(dir):
        if f.lower().endswith('.pdf'):
            os.remove(os.path.join(dir, f))
    print("======미국 완료=====")
    return 'result_us.csv'
