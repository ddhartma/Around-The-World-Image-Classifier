from django.shortcuts import render
from django.conf import settings

from photos.views import photo_list_classification
from classification.views import check_folderPaths

import requests
import os
import subprocess
from applescript import tell
from shutil import copyfile
from datetime import datetime, date, timedelta

from django.http import HttpResponse 
import pandas as pd
from collections import Counter
import collections

import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)

from math import radians, cos, sin, asin, sqrt

def results_dataframe():
    # Get result folderpath
    image_folder, yolo_folder, person_folder, file_path_copy_dataframe = check_folderPaths()
    result_path = image_folder + '.txt'
    result_folder, _ = os.path.split(result_path)
    _, result_file_name = os.path.split(image_folder)

    # df as xlsx file
    result_xlsx_file = result_file_name + '.xlsx'
    result_xlsx_path = image_folder + '.xlsx'
    
    # df_count as xlsx
    result_imgPerDay_xlsx_file = result_file_name + '_imgPerDay.xlsx'
    result_imgPerDay_xlsx_path = image_folder + '_imgPerDay.xlsx'


    # df as html
    result_html_file = result_file_name + '.html'
    result_html_path = image_folder + '.html'

    # Build paths inside the project like this: os.path.join(BASE_DIR, ...)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dst_path = os.path.join(BASE_DIR, 'templates', 'table.html')
    print('dst_path')
    print(dst_path)
    copyfile(result_html_path, dst_path)

    return result_xlsx_path, result_imgPerDay_xlsx_path, image_folder

def dataframe_analysis():
    result_xlsx_path, result_imgPerDay_xlsx_path , image_folder = results_dataframe()

    print('---------------------------------')
    print('result_xlsx_path:')
    print(result_xlsx_path)
    print('result_imgPerDay_xlsx_path:')
    print(result_imgPerDay_xlsx_path)
    print('---------------------------------')
    df = pd.read_excel(result_xlsx_path)
    df_count = pd.read_excel(result_imgPerDay_xlsx_path)
    print(df.describe())
    print(df.head(10))

    # yolo classifications
    yolo_list = [eval(x) for x in df['classes_yolo'].to_list()]
    yolo_flat_list = [item for sublist in yolo_list for item in sublist]
    yolo_flat_list = list(set(yolo_flat_list))
    print(yolo_flat_list)

    # ImageNet classifications
    imageNet_list = [eval(x) for x in df['classes_ImgNet'].to_list()]
    imageNet_flat_list = [item for sublist in imageNet_list for item in sublist]
    imageNet_flat_list = list(set(imageNet_flat_list))
    print(imageNet_flat_list)

    return yolo_flat_list, imageNet_flat_list




def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

# Get  data from website: Yolo, ImageNet, 
def filter_results(request): 
    
    yolo_flat_list, imageNet_flat_list = dataframe_analysis()
    result_xlsx_path, result_imgPerDay_xlsx_path , image_folder = results_dataframe()

    yolo_flat_list.insert(0,'no class')
    imageNet_flat_list.insert(0,'no class')
    yolo_flat_list = [element.strip().replace(' ', '_') for element in yolo_flat_list]
    imageNet_flat_list = [element.strip().replace(' ', '_') for element in imageNet_flat_list]



    if request.method == 'POST':
        selected_value_yolo = request.POST['drop1']
        selected_value_imageNet = request.POST['drop2']
        selected_start_date = request.POST['start']
        selected_end_date = request.POST['end']
        
        current_location_lat = request.POST['current_location_lat']
        current_location_lon = request.POST['current_location_lon']
        

        current_radius = request.POST['current_radius']

        print('selected_value_yolo')
        print(selected_value_yolo)
        print('selected_value_imageNet')
        print(selected_value_imageNet)
        print('selected_start_date')
        print(selected_start_date)
        print('selected_end_date')
        print(selected_end_date)

        print('current_location_lat')
        print(current_location_lat)
        print('current_location_lon')
        print(current_location_lon)
        print('current_radius')
        print(current_radius)

        ctx = {
            'yolo_select': yolo_flat_list,
            'imageNet_select': imageNet_flat_list,
            'selected_yolo': selected_value_yolo,
            'selected_imageNet': selected_value_imageNet,
            'selected_start': selected_start_date,
            'selected_end': selected_end_date,
            'current_loc_lat': current_location_lat,
            'current_loc_lon': current_location_lon,
            'current_rad': current_radius,
            }
       
    else:
        print('DA BIN ICH AUCH DURCH')
        ctx = {
            'yolo_select': yolo_flat_list,
            'imageNet_select': imageNet_flat_list,
            'selected_yolo': '',
            'selected_imageNet': '',
            'selected_start': datetime.today().strftime('%m/%d/%Y'),
            'selected_end':  datetime.today().strftime('%m/%d/%Y'),
            'current_loc_lat': '',
            'current_loc_lon': '',
            'current_rad': '',
            }
       
    return render(request, 'f_filter.html', ctx)
    

def imageNet_class_object(request): 
   
    answer = request.GET['imageNet_dropdown'] 
    print(answer)