# Generated by Django 3.2.6 on 2022-08-18 05:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('baekho', '0014_headoffice_img'),
    ]

    operations = [
        migrations.AddField(
            model_name='headoffice',
            name='country',
            field=models.CharField(default=' ', max_length=10),
        ),
    ]