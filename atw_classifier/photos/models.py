from __future__ import unicode_literals
from django.db import models

from PIL import Image 

from io import BytesIO

# Create your models here.
class Photo(models.Model):
    upload_folder = ''
    title = models.CharField(max_length=200)
    width = models.IntegerField(default=0)
    height = models.IntegerField(default=0)
    file = models.FileField(upload_to=upload_folder)
    image = models.ImageField(upload_to=upload_folder, null=False,blank=False, width_field="width", height_field="height")
    timestamp = models.DateTimeField(auto_now_add=True, auto_now=False)
    #updated = models.DateTimeField(auto_now_add=False,auto_now=True)

    def __unicode__(self):
        return self.title

    def rotate_save(self, *args, **kwargs):
        if self.image:
            pilImage = Image.open(BytesIO(self.image.read()))
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(pilImage._getexif().items())

            if exif[orientation] == 3:
                pilImage = pilImage.rotate(180, expand=True)
            elif exif[orientation] == 6:
                pilImage = pilImage.rotate(270, expand=True)
            elif exif[orientation] == 8:
                pilImage = pilImage.rotate(90, expand=True)

            output = BytesIO()
            pilImage.save(output, format='JPEG', quality=75)
            output.seek(0)
            self.image = File(output, self.image.name)

            print('rotate triggered')

        return super(Photo, self).save(*args, **kwargs)


class Photo_yolo(models.Model):
    upload_folder = ''
    title = models.CharField(max_length=200)
    width = models.IntegerField(default=0)
    height = models.IntegerField(default=0)
    file = models.FileField(upload_to=upload_folder)
    image = models.ImageField(upload_to=upload_folder, null=False,blank=False, width_field="width", height_field="height")
    timestamp = models.DateTimeField(auto_now_add=True, auto_now=False)
    #updated = models.DateTimeField(auto_now_add=False,auto_now=True)
    



