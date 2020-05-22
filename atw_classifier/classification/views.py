from django.shortcuts import render
from django.conf import settings

from photos.views import photo_list_classification
from django.contrib.auth.decorators import login_required

import requests
import os
import subprocess
try:
    from applescript import tell
except:
    pass

def check_folderPaths():
    cwd_path = os.getcwd()
    file_path = os.path.join(cwd_path, 'templates', 'file_path.txt')

    try:
        with open(file_path, 'r') as f:
            file_path_str =f.read()
    except:
        pass

    try:
        image_folder = file_path_str.split(',')[0]
    except:
        image_folder = 'No image_folder found'

    try:
        yolo_folder = file_path_str.split(',')[1]
    except:
        yolo_folder = 'No yolo_folder found'

    try:
        person_folder = file_path_str.split(',')[2]
    except:
        person_folder = 'No person_folder found'

    try:
        file_path_copy_dataframe = file_path_str.split(',')[3]
    except:
        file_path_copy_dataframe = 'No file_path_copy_dataframe found'

    return image_folder, yolo_folder, person_folder, file_path_copy_dataframe

@login_required
def classification(request):

    image_folder, yolo_folder, person_folder, file_path_copy_dataframe = check_folderPaths()
    photo_context = photo_list_classification()

    return render(request, 'classification.html', {'data1': image_folder,
                                             'data2': yolo_folder,
                                             'data3': person_folder,
                                             'data4': file_path_copy_dataframe,
                                             'photos_class': photo_context['photos'],
                                             })
@login_required
def jupy_nb(request):
    """
    openJupyter = 'jupyter notebook'
    cwd_path = os.getcwd()
    path = os.path.join(cwd_path, 'around_the_world_classifier.ipynb')
    subprocess.Popen(openJupyter + path.replace(' ', '\ '), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,stdin=subprocess.PIPE)
    """

    image_folder, yolo_folder, person_folder, file_path_copy_dataframe = check_folderPaths()
    photo_context = photo_list_classification()

    cwd_path = os.getcwd()

    try:
        path = os.path.join(cwd_path, 'templates').replace('\\', '/').replace(' ', '\\')
        yourCommand = 'conda activate base && ' + 'cd ' + path + ' && ' + 'jupyter notebook around_the_world_classifier.ipynb'

        tell.app( 'Terminal', 'do script "' + yourCommand + '"')

    except:

        path = os.path.join(cwd_path, 'templates', 'around_the_world_classifier.ipynb')
        openJupyter = "conda activate MTP_LSTM && jupyter notebook " + path
        subprocess.call(openJupyter, shell=True)


    return render(request, 'classification.html',{'data1': image_folder,
                                             'data2': yolo_folder,
                                             'data3': person_folder,
                                             'data4': file_path_copy_dataframe,
                                             'photos_class': photo_context['photos'],
                                             })

@login_required
def home(request):
    return render(request, 'e_info.html')

@login_required
def results(request):
    return render(request, 'c_results.html')


@login_required
def filter(request):
    return render(request, 'd_filter.html')

@login_required
def info(request):
    return render(request, 'e_info.html')
