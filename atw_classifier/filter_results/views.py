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
import math

import itertools
import numpy as np

from PIL import Image, ImageDraw
import base64
import io


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

# Filter for datetimes
def time_periods_eval(df_in, start_datetime , end_datetime, time_periods = True):
    if time_periods == True:
        datetime_filter_lower = df_in['date_time'] >= start_datetime
        datetime_filter_upper = df_in['date_time'] <= end_datetime

        # Select all cases where df['date_time'] >= start_datetime and df['date_time'] <= end_datetime
        df_filter = df_in[datetime_filter_lower & datetime_filter_upper]
        df_filter.sort_values('date_time', inplace=True, ascending=True)
    else:
        df_filter = df_in
    return df_filter

# Filter Yolo, ImageNet
def filter_by_groups_eval(df_in, selected_value_yolo, selected_value_imageNet, selected_logic='AND', filter_by_groups = True):
    if filter_by_groups == True:
        classes_yolo = [eval(x) for x in df_in['classes_yolo'].to_list()]
        classes_yolo_merged = list(itertools.chain(*classes_yolo))
        df_in['yolo_chosen'] = np.nan
        df_in['yolo_chosen'] = [selected_value_yolo in j for j in classes_yolo]
       
        classes_ImgNet = [eval(x) for x in df_in['classes_ImgNet'].to_list()]
        classes_ImgNet_merged = list(itertools.chain(*classes_ImgNet))
        df_in['ImgNet_chosen'] = np.nan
        df_in['ImgNet_chosen'] = [selected_value_imageNet in j for j in classes_ImgNet]
        
        if selected_value_yolo !='' and selected_value_imageNet =='':
            if selected_value_yolo in classes_yolo_merged:
                df_filter = df_in[df_in.yolo_chosen]
            else:
                df_filter = df_in
            
        if selected_value_yolo =='' and selected_value_imageNet !='':
            if selected_value_imageNet in classes_ImgNet_merged:
                df_filter = df_in[df_in.ImgNet_chosen]
            else:
                df_filter = df_in
              
        if selected_value_yolo !='' and selected_value_imageNet !='':
            if selected_value_yolo in classes_yolo_merged and selected_value_imageNet in classes_ImgNet_merged:
                print('case1')
                if selected_logic == 'OR':
                    df_filter = df_in[df_in.yolo_chosen | df_in.ImgNet_chosen]
                elif selected_logic == 'AND':
                    df_filter = df_in[df_in.yolo_chosen & df_in.ImgNet_chosen]
            elif selected_value_yolo not in classes_yolo_merged and selected_value_imageNet in classes_ImgNet_merged:
                print('case2')
                df_filter = df_in[df_in.ImgNet_chosen]
            elif selected_value_yolo in classes_yolo_merged and selected_value_imageNet not in classes_ImgNet_merged:
                print('case3')
                df_filter = df_in[df_in.yolo_chosen]
            else:
                print('case4')
                df_filter = df_in
            
        df_filter.sort_values('date_time', inplace=True, ascending=True)
        
    else: 
        df_filter = df_in
    return df_filter
        

# CHeck if test GPS coordinates are within a circle with given lon, lat and radius
def haversine(df_in, lon1, lat1, radius, gps_areas=True):
    if gps_areas == True:
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        print('df_in[GPS]')
        print(df_in['GPS'])

        df_in['lat2'] = [eval(x)[0] for x in df_in['GPS'].to_list()]
        df_in['lon2'] = [eval(x)[1] for x in df_in['GPS'].to_list()]
        
        lat2 = df_in['lat2']
        lon2 = df_in['lon2']


        print('lat2 in')
        print(lat2)
        print('lon2 in')
        print(lon2)
    
        # convert decimal degrees to radians 
        lon1, lat1 = map(radians, [float(lon1), float(lat1)])

        lon2 = lon2.astype(float)
        lat2 = lat2.astype(float)
        lon2 = np.radians(lon2)
        lat2 = np.radians(lat2)

        print('lon1 radians')
        print(lon1)
        print('lat1 radians')
        print(lat1)

        print('lon2 radians')
        print(lon2)
        print('lat2 radians')
        print(lat2)

        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 

        print('dlon')
        print(dlon)
        print('dlat')
        print(dlat)


        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a)) 
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles

        df_filter = df_in[c*r <= float(radius)]
        return df_filter

    else: 
        df_filter = df_in
    return df_filter


# Transform DataFRame with images to HTML
#  - Check  file path. If null return False (to get None for Byte conversion)
#  - Convert image to thumbnail
#  - return image as raw Base64
def check_file_path(row):
    try:
        return os.path.isfile(row)
    except:
        return False

def get_thumbnail(path):
   
    i = Image.open(path)
    i.thumbnail((300, 300), Image.LANCZOS)
    return i

def get_image_asraw_Base64(im):
    buffer = io.BytesIO()
    im.save(buffer, format='PNG')
    buffer.seek(0)

    data_uri = base64.b64encode(buffer.read()).decode('ascii')

    #html = '<html><head></head><body>'
    html_img = '<img src="data:image/png;base64,{0}">'.format(data_uri)
    #html += '</body></html>'

    return html_img
    

# Transform DataFrame with images to HTML
def df_as_html(df):
    pd.set_option('display.max_colwidth', -1) 
    
    df['image'] = df.apply(lambda row: get_thumbnail(row['img_path']), axis = 1)
    df['exists'] = df.apply(lambda row: check_file_path(row['img_path_yolo']), axis = 1)

    df['image_yolo'] = df.apply(lambda row: get_thumbnail(row['img_path_yolo']) if row['exists'] == True else None, axis = 1)
   
    try: 
        df.drop('exists', inplace =True, axis=1)
    except:
        pass

    
    # Convert DataFrame to HTML, displaying PIL.Image objects embedded in dataframe
    html = df.to_html(formatters={'image': get_image_asraw_Base64, 'image_yolo': get_image_asraw_Base64}, escape=False)
   
    #with open(image_folder + '.html', 'w') as f:
    with open('test.html', 'w') as f:
        f.write(html)

def get_filtered_dataFrame(selected_start_date, selected_end_date, selected_value_yolo, selected_value_imageNet, selected_logic, current_location_lat, current_location_lon, current_radius):
    result_xlsx_path, result_imgPerDay_xlsx_path , image_folder = results_dataframe()
    df = pd.read_excel(result_xlsx_path)

    try: 
        df.drop('Unnamed: 0', inplace =True, axis=1)
    except:
        pass

    # datatime filter
    df_filter = time_periods_eval(df, selected_start_date, selected_end_date, time_periods = True)
    print('df_filter datetime')
    print(df_filter)

    # yolo & imageNet filter
    df_filter = filter_by_groups_eval(df_filter, selected_value_yolo, selected_value_imageNet, selected_logic, filter_by_groups = True)
    print('df_filter classes')
    print(df_filter)

    # GPS filter
    df_filter = haversine(df_filter, current_location_lon, current_location_lat, current_radius, gps_areas=True)

    print('df_filter gps')
    print(df_filter)



    df_as_html(df_filter)

# Get data choice from website: Yolo, ImageNet, datetime selction, GPS data
def filter_results(request): 
    
    yolo_flat_list, imageNet_flat_list = dataframe_analysis()
    result_xlsx_path, result_imgPerDay_xlsx_path , image_folder = results_dataframe()

    yolo_flat_list.insert(0,'all')
    imageNet_flat_list.insert(0,'all')
    yolo_flat_list = [element.strip().replace(' ', '_') for element in yolo_flat_list]
    imageNet_flat_list = [element.strip().replace(' ', '_') for element in imageNet_flat_list]

    
    if request.method == 'POST':
        selected_value_yolo = request.POST['drop1']
        selected_value_imageNet = request.POST['drop2']

        selected_logic = request.POST.get('selected_logic_sl', False)

        selected_start_date = request.POST['start']
        selected_end_date = request.POST['end']
        
        current_location_lat = request.POST['current_location_lat']
        current_location_lon = request.POST['current_location_lon']
        

        current_radius = request.POST['current_radius']

        print('selected_value_yolo')
        print(selected_value_yolo)
        print('selected_value_imageNet')
        print(selected_value_imageNet)

        print('selected_logic')
        print(selected_logic)

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

        if selected_logic == 'on':
            selected_logic_label = 'OR'
            selected_logic_back = 'checked'
        else:
            selected_logic_label = 'AND'
            selected_logic_back =''

        ctx = {
            'yolo_select': yolo_flat_list,
            'imageNet_select': imageNet_flat_list,
            'selected_yolo': selected_value_yolo,
            'selected_imageNet': selected_value_imageNet,
            'selec_log': selected_logic_back,
            'selected_start': selected_start_date,
            'selected_end': selected_end_date,
            'current_loc_lat': current_location_lat,
            'current_loc_lon': current_location_lon,
            'current_rad': current_radius,
            }

        

        photo_list_to_display = get_filtered_dataFrame(selected_start_date, 
                                                       selected_end_date, 
                                                       selected_value_yolo,
                                                       selected_value_imageNet,
                                                       selected_logic_label,
                                                       current_location_lat,
                                                       current_location_lon,
                                                       current_radius
                                                       )

       
    else:
        print('DA BIN ICH AUCH DURCH')
        ctx = {
            'yolo_select': yolo_flat_list,
            'imageNet_select': imageNet_flat_list,
            'selected_yolo': '',
            'selected_imageNet': '',
            'selec_log': 'checked',
            'selected_start': '01/01/1976',
            'selected_end':  datetime.today().strftime('%m/%d/%Y'),
            'current_loc_lat': '',
            'current_loc_lon': '',
            'current_rad': '',
            }
       
    return render(request, 'f_filter.html', ctx)
    

def imageNet_class_object(request): 
   
    answer = request.GET['imageNet_dropdown'] 
    print(answer)