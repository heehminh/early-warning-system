# Generated by Django 3.2.6 on 2022-08-12 02:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('baekho', '0006_aumodel_chinamodel_japanmodel_usamodel_vtmodel'),
    ]

    operations = [
        migrations.AddField(
            model_name='vtmodel',
            name='word1_code',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word1_name',
            field=models.CharField(default='NaN', max_length=100),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word1_ngram',
            field=models.CharField(default='NaN', max_length=100),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word1_sim',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word2_code',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word2_name',
            field=models.CharField(default='NaN', max_length=100),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word2_ngram',
            field=models.CharField(default='NaN', max_length=100),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word2_sim',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word3_code',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word3_name',
            field=models.CharField(default='NaN', max_length=100),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word3_ngram',
            field=models.CharField(default='NaN', max_length=100),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word3_sim',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word4_code',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word4_name',
            field=models.CharField(default='NaN', max_length=100),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word4_ngram',
            field=models.CharField(default='NaN', max_length=100),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word4_sim',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word5_code',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word5_name',
            field=models.CharField(default='NaN', max_length=100),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word5_ngram',
            field=models.CharField(default='NaN', max_length=100),
        ),
        migrations.AddField(
            model_name='vtmodel',
            name='word5_sim',
            field=models.FloatField(default=0),
        ),
        migrations.AlterField(
            model_name='vtmodel',
            name='date',
            field=models.CharField(max_length=50),
        ),
    ]