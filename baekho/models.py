from django.db import models

# Create your models here.

class Country(models.Model):
    name = models.CharField(max_length=50)
    name_eng = models.CharField(max_length=60, default="")
    img_path = models.URLField(default="")
    txt = models.CharField(max_length=200, default="고시정보 출처 설명")

    def __str__(self): # 이 메뉴클래스를 하나의 문자열로 만들어줌
        return self.name
    
# 오프닝 화면에서 최근 고시정보 5개를 보여주는 모델
class Opening(models.Model):
    title = models.CharField(max_length=100) # 관세청 타이틀
    date = models.DateField()
    url = models.URLField()
    
    def __str__(self):
        return self.title

class Industry(models.Model):
    number = models.IntegerField(default=0)
    hscode = models.CharField(max_length=10)
    hscode_name = models.CharField(max_length=50)
    KSIC10 = models.CharField(max_length=10)
    KSIC10_name = models.CharField(max_length=50)

    def __str__(self):
        return self.hscode_name

class Compare(models.Model):
    last_china = models.CharField(max_length=200)
    last_usa = models.CharField(max_length=200)
    last_japan = models.CharField(max_length=200)
    last_vietnam = models.CharField(max_length=200)
    last_australia = models.CharField(max_length=200)

    def __str__(self):
        return self.last_china

# 중국
class HeadOffice(models.Model):
    number = models.IntegerField(default=0)
    title = models.CharField(max_length=100) # 관세청 타이틀
    date = models.CharField(max_length=100)
    url = models.URLField()
    img = models.ImageField(null=True)
    country = models.CharField(max_length=10, default=" ")
    country_eng = models.CharField(max_length=10, default=" ")
    korea = models.BooleanField(default=False)
    export = models.BooleanField(default=False)
    
    word1_ngram = models.CharField(max_length=100, default="NaN")
    word1_code = models.CharField(max_length=100, default="NaN")
    word1_name = models.CharField(max_length=100, default="NaN")
    word1_sim = models.CharField(max_length=100, default="NaN")

    word2_ngram = models.CharField(max_length=100, default="NaN")
    word2_code = models.CharField(max_length=100, default="NaN")
    word2_name = models.CharField(max_length=100, default="NaN")
    word2_sim = models.CharField(max_length=100, default="NaN")

    word3_ngram = models.CharField(max_length=100, default="NaN")
    word3_code = models.CharField(max_length=100, default="NaN")
    word3_name = models.CharField(max_length=100, default="NaN")
    word3_sim = models.CharField(max_length=100, default="NaN")

    word4_ngram = models.CharField(max_length=100, default="NaN")
    word4_code = models.CharField(max_length=100, default="NaN")
    word4_name = models.CharField(max_length=100, default="NaN")
    word4_sim = models.CharField(max_length=100, default="NaN")
    
    txt_eng = models.CharField(max_length=1000, default=" ")
    txt = models.CharField(max_length=1000, default=" ")

    def __str__(self):
        return self.title