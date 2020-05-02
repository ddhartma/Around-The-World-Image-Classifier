from __future__ import unicode_literals
from django.db import models
import os

# Create your models here.
class Photo(models.Model):
    """
    cwd_path = os.getcwd()
    file_path = os.path.join(cwd_path, 'templates', 'file_path.txt')

    try:
        with open(file_path, 'r') as f:
            file_path_str =f.read()
    except:
        pass
            
    try: 
        image_folder = file_path_str.split(',')[0]
        result_path = image_folder + '.txt'
        result_folder, _ = os.path.split(result_path)

    except:
        result_folder = ''
    
    """
   

    title = models.CharField(max_length=200)
    width = models.IntegerField(default=0)
    height = models.IntegerField(default=0)
    file = models.FileField(upload_to='')
    image = models.ImageField(upload_to='', null=False,blank=False, width_field="width", height_field="height")
    timestamp = models.DateTimeField(auto_now_add=True, auto_now=False)
    #updated = models.DateTimeField(auto_now_add=False,auto_now=True)
    

    def __unicode__(self):
        return self.title

    class Meta:
        ordering = ["-timestamp"]

