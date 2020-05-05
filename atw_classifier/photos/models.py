from __future__ import unicode_literals
from django.db import models
import os

# Create your models here.
class Photo(models.Model):
    upload_folder = 'weltreise_2014'
    title = models.CharField(max_length=200)
    width = models.IntegerField(default=0)
    height = models.IntegerField(default=0)
    file = models.FileField(upload_to=upload_folder)
    image = models.ImageField(upload_to=upload_folder, null=False,blank=False, width_field="width", height_field="height")
    timestamp = models.DateTimeField(auto_now_add=True, auto_now=False)
    #updated = models.DateTimeField(auto_now_add=False,auto_now=True)
    

    def __unicode__(self):
        return self.title

    class Meta:
        ordering = ["-timestamp"]

