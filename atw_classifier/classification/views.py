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
    # Build paths inside the project like this: os.path.join(BASE_DIR, ...)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # IMAGES
    try:
        # Path images
        image_folder = os.path.join(os.path.dirname(BASE_DIR), "images")
        # Create images_yolo directory 
        os.makedirs(image_folder) 
    except: 
        pass

    # YOLO
    try:
        # Path yolo images
        yolo_folder = os.path.join(os.path.dirname(BASE_DIR), "images_yolo")
        # Create images_yolo directory 
        os.makedirs(yolo_folder) 
    except: 
        pass

    # PERSON
    try:
        # Path personal images
        person_folder = os.path.join(os.path.dirname(BASE_DIR), "images_personal")
        
        # Create images_personal directory 
        os.makedirs(person_folder) 
    except: 
        pass

    # BACKUP
    try:
        # Path backup
        file_path_copy_dataframe = os.path.join(os.path.dirname(BASE_DIR), "backup")
        
        # Create backup directory 
        os.makedirs(file_path_copy_dataframe) 
    except: 
        pass

    
    file_path_str = image_folder + ',' + yolo_folder + ',' + person_folder + ',' + file_path_copy_dataframe
    with open(file_path, 'w') as f:
        f.write(file_path_str)

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
        print(path)
        yourCommand = 'conda activate base && ' + 'cd ' + path + ' && ' + 'jupyter notebook around_the_world_classifier.ipynb'

        tell.app( 'Terminal', 'do script "' + yourCommand + '"')
        print('Terminal started')
    except Exception as e:
        print(e)

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
