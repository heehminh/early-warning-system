from django.db import models

# Create your models here.

class Country(models.Model):
    name = models.CharField(max_length=50)

    def __str__(self): # 이 메뉴클래스를 하나의 문자열로 만들어줌
        return self.name