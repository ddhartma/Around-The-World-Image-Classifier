# Generated by Django 3.0.5 on 2020-05-01 09:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('photos', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='photo',
            name='file',
            field=models.FileField(default='', upload_to='photos/'),
            preserve_default=False,
        ),
    ]
