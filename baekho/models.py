from django.db import models

# Create your models here.

class Country(models.Model):
    name = models.CharField(max_length=50)
    name_eng = models.CharField(max_length=60, default="")

    def __str__(self): # 이 메뉴클래스를 하나의 문자열로 만들어줌
        return self.name
    
# 오프닝 화면에서 최근 고시정보 5개를 보여주는 모델 
class Opening(models.Model):
    title = models.CharField(max_length=100) # 관세청 타이틀
    date = models.DateField()
    url = models.URLField()
    
    def __str__(self):
        return self.title

# 중국
class ChinaModel(models.Model):
    title = models.CharField(max_length=100) # 관세청 타이틀
    date = models.DateField()
    url = models.URLField()
    
    def __str__(self):
        return self.title

# 미국 
class USAModel(models.Model):
    title = models.CharField(max_length=100) # 관세청 타이틀
    date = models.DateField()
    url = models.URLField()
    
    def __str__(self):
        return self.title

# 일본
class JapanModel(models.Model):
    title = models.CharField(max_length=100) # 관세청 타이틀
    date = models.DateField()
    url = models.URLField()
    
    def __str__(self):
        return self.title

# 베트남
class VtModel(models.Model):
    title = models.CharField(max_length=100) # 관세청 타이틀
    date = models.CharField(max_length=20)
    url = models.URLField()
    
    def __str__(self):
        return self.title

# 호주 
class AuModel(models.Model):
    title = models.CharField(max_length=100) # 관세청 타이틀
    date = models.DateField()
    url = models.URLField()
    
    def __str__(self):
        return self.title