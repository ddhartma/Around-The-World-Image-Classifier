from django.shortcuts import render
from .models import Photo
# Create your views here.

def photo_list(request):
    queryset = Photo.objects.all()
    context = {'photos': queryset,
    }
    return render(request, "c_photo.html", context)


def photo_list_classification():
    queryset = Photo.objects.all()
    photo_context = {'photos': queryset,
    }
    #return render(request, "b_classification.html", context)
    return photo_context
