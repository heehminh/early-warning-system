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

# 중국
class HeadOffice(models.Model):
    number = models.IntegerField(default=0)
    title = models.CharField(max_length=100) # 관세청 타이틀
    date = models.CharField(max_length=100)
    url = models.URLField()
    img = models.ImageField(null=True)
    country = models.CharField(max_length=10, default=" ")

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
    
    def __str__(self):
        return self.title

# # 중국
# class ChinaModel(models.Model):
#     title = models.CharField(max_length=100) # 관세청 타이틀
#     date = models.DateField()
#     url = models.URLField()

#     word1_ngram = models.CharField(max_length=100, default="NaN")
#     word1_code = models.CharField(max_length=100, default="NaN")
#     word1_name = models.CharField(max_length=100, default="NaN")
#     word1_sim = models.CharField(max_length=100, default="NaN")

#     word2_ngram = models.CharField(max_length=100, default="NaN")
#     word2_code = models.CharField(max_length=100, default="NaN")
#     word2_name = models.CharField(max_length=100, default="NaN")
#     word2_sim = models.CharField(max_length=100, default="NaN")

#     word3_ngram = models.CharField(max_length=100, default="NaN")
#     word3_code = models.CharField(max_length=100, default="NaN")
#     word3_name = models.CharField(max_length=100, default="NaN")
#     word3_sim = models.CharField(max_length=100, default="NaN")

#     word4_ngram = models.CharField(max_length=100, default="NaN")
#     word4_code = models.CharField(max_length=100, default="NaN")
#     word4_name = models.CharField(max_length=100, default="NaN")
#     word4_sim = models.CharField(max_length=100, default="NaN")
    
#     def __str__(self):
#         return self.title

# # 미국 
# class USAModel(models.Model):
#     title = models.CharField(max_length=100) # 관세청 타이틀
#     date = models.DateField()
#     url = models.URLField()
    
#     def __str__(self):
#         return self.title

# # 일본
# class JapanModel(models.Model):
#     title = models.CharField(max_length=100) # 관세청 타이틀
#     date = models.DateField()
#     url = models.URLField()
    
#     def __str__(self):
#         return self.title

# # 베트남
# class VtModel(models.Model):
#     title = models.CharField(max_length=100) # 관세청 타이틀
#     date = models.CharField(max_length=50)
#     url = models.URLField()

#     word1_ngram = models.CharField(max_length=100, default="NaN")
#     word1_code = models.CharField(max_length=100, default="NaN")
#     word1_name = models.CharField(max_length=100, default="NaN")
#     word1_sim = models.CharField(max_length=100, default="NaN")

#     word2_ngram = models.CharField(max_length=100, default="NaN")
#     word2_code = models.CharField(max_length=100, default="NaN")
#     word2_name = models.CharField(max_length=100, default="NaN")
#     word2_sim = models.CharField(max_length=100, default="NaN")

#     word3_ngram = models.CharField(max_length=100, default="NaN")
#     word3_code = models.CharField(max_length=100, default="NaN")
#     word3_name = models.CharField(max_length=100, default="NaN")
#     word3_sim = models.CharField(max_length=100, default="NaN")

#     word4_ngram = models.CharField(max_length=100, default="NaN")
#     word4_code = models.CharField(max_length=100, default="NaN")
#     word4_name = models.CharField(max_length=100, default="NaN")
#     word4_sim = models.CharField(max_length=100, default="NaN")
 
#     def __str__(self):
#         return self.title

# # 호주 
# class AuModel(models.Model):
#     title = models.CharField(max_length=100) # 관세청 타이틀
#     date = models.DateField()
#     url = models.URLField()
    
#     def __str__(self):
#         return self.title