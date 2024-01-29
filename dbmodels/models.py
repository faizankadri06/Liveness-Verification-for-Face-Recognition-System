from django.db import models
class Login(models.Model):
    username = models.CharField(primary_key=True, max_length=50)
    password = models.CharField(max_length=50)

    loginauth_objects=models.Manager()

    class Meta:
        managed = False
        db_table = 'login'


class ALogin(models.Model):
    username = models.CharField(primary_key=True, max_length=50)
    password = models.CharField(max_length=50)

    loginauth_objects=models.Manager()

    class Meta:
        managed = False
        db_table = 'a_login'


class Datim(models.Model):
    plogin = models.DateTimeField(primary_key=True)
    plogout = models.DateTimeField(blank=True, null=True)
    uid = models.ForeignKey('User', models.DO_NOTHING, db_column='uid', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'datim'


class User(models.Model):
    uid = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=100, blank=True, null=True)
    email = models.CharField(max_length=50, blank=True, null=True)
    phone = models.BigIntegerField(blank=True, null=True)
    username = models.ForeignKey(Login, models.DO_NOTHING, db_column='username', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'user'