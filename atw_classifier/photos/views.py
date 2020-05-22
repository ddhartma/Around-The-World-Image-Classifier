import time
from django.shortcuts import render, redirect
from .models import Photo, Photo_yolo


from django.http import JsonResponse
from django.views import View
from .forms import PhotoForm

from django.contrib.auth.decorators import login_required

# Create your views here.

class BasicUploadView(View):
    @login_required
    def get(self, request):
        photos_list = Photo.objects.all()
        return render(self.request, 'photos/basic_upload/index.html', {'photos': photos_list})

    @login_required
    def post(self, request):
        form = PhotoForm(self.request.POST, self.request.FILES)
        if form.is_valid():
            photo = form.save()
            data = {'is_valid': True, 'name': photo.file.name, 'url': photo.file.url}
        else:
            data = {'is_valid': False}
        return JsonResponse(data)

class ProgressBarUploadView(View):
    @login_required
    def get(self, request):
        photos_list = Photo.objects.all()
        return render(self.request, 'photos/progress_bar_upload/index.html', {'photos': photos_list})

    @login_required
    def post(self, request):
        time.sleep(1)  # You don't need this line. This is just to delay the process so you can see the progress bar testing locally.
        form = PhotoForm(self.request.POST, self.request.FILES)
        if form.is_valid():
            photo = form.save()
            data = {'is_valid': True, 'name': photo.file.name, 'url': photo.file.url}
        else:
            data = {'is_valid': False}
        return JsonResponse(data)


class DragAndDropUploadView(View):
    @login_required
    def get(self, request):
        photos_list = Photo.objects.all()
        return render(self.request, 'photos/drag_and_drop_upload/index.html', {'photos': photos_list})

    @login_required
    def post(self, request):
        form = PhotoForm(self.request.POST, self.request.FILES)
        if form.is_valid():
            photo = form.save()
            data = {'is_valid': True, 'name': photo.file.name, 'url': photo.file.url}
        else:
            data = {'is_valid': False}
        return JsonResponse(data)

@login_required
def clear_database(request):
    for photo in Photo.objects.all():
        photo.file.delete()
        photo.delete()
    return redirect(request.POST.get('next'))


@login_required
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


def photo_yolo_list_classification():
    queryset = Photo_yolo.objects.all()
    photo_context_yolo = {'photos_yolo': queryset,
    }
    return photo_context_yolo


