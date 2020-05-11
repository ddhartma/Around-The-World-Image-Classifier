from django.shortcuts import render
from django.conf import settings

from photos.views import photo_list_classification
from photos.models import Photo
from classification.views import check_folderPaths

import requests
import os
import subprocess
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

import json

# Get folder paths and file names for DataFrames df, copy html report
def results_dataframe():
    # Get result folderpath
    image_folder, yolo_folder, person_folder, file_path_copy_dataframe = check_folderPaths()
    result_path = image_folder + '.txt'
    result_folder, _ = os.path.split(result_path)
    _, result_file_name = os.path.split(image_folder)

    # df as xlsx file
    result_xlsx_file = result_file_name + '.xlsx'
    result_xlsx_path = image_folder + '.xlsx'

    return result_xlsx_path, image_folder

def dataframe_analysis():
    result_xlsx_path, _ = results_dataframe()

    df = pd.read_excel(result_xlsx_path)

    # yolo classifications
    yolo_list = [eval(x) for x in df['classes_yolo'].to_list()]
    yolo_flat_list = [item for sublist in yolo_list for item in sublist]
    yolo_flat_list = list(set(yolo_flat_list))


    # ImageNet classifications
    imageNet_list = [eval(x) for x in df['classes_ImgNet'].to_list()]
    imageNet_flat_list = [item for sublist in imageNet_list for item in sublist]
    imageNet_flat_list = list(set(imageNet_flat_list))

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
                if selected_logic == 'OR':
                    df_filter = df_in[df_in.yolo_chosen | df_in.ImgNet_chosen]
                elif selected_logic == 'AND':
                    df_filter = df_in[df_in.yolo_chosen & df_in.ImgNet_chosen]
            elif selected_value_yolo not in classes_yolo_merged and selected_value_imageNet in classes_ImgNet_merged:
                df_filter = df_in[df_in.ImgNet_chosen]
            elif selected_value_yolo in classes_yolo_merged and selected_value_imageNet not in classes_ImgNet_merged:
                df_filter = df_in[df_in.yolo_chosen]
            else:
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
        df_in['lat2'] = [eval(x)[0] for x in df_in['GPS'].to_list()]
        df_in['lon2'] = [eval(x)[1] for x in df_in['GPS'].to_list()]

        lat2 = df_in['lat2']
        lon2 = df_in['lon2']

        # convert decimal degrees to radians
        lon1, lat1 = map(radians, [float(lon1), float(lat1)])

        lon2 = lon2.astype(float)
        lat2 = lat2.astype(float)
        lon2 = np.radians(lon2)
        lat2 = np.radians(lat2)

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles

        df_filter = df_in[c*r <= float(radius)]

        return df_filter

    else:

        return df_in


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
    with open('./templates/filter_result.html', 'w') as f:
        f.write(html)

# Collector of function calls to get df_filter
def get_filtered_dataFrame(selected_start_date, selected_end_date, selected_value_yolo, selected_value_imageNet, selected_logic, current_location_lat, current_location_lon, current_radius, selected_gps_label):
    result_xlsx_path, image_folder = results_dataframe()
    df = pd.read_excel(result_xlsx_path)
    if df.empty == False:
        try:
            df.drop('Unnamed: 0', inplace =True, axis=1)
        except:
            pass

        # datatime filter
        df_filter = time_periods_eval(df, selected_start_date, selected_end_date, time_periods = True)

        if df_filter.empty == False:
            # yolo & imageNet filter
            df_filter = filter_by_groups_eval(df_filter, selected_value_yolo, selected_value_imageNet, selected_logic, filter_by_groups = True)

        if df_filter.empty == False:
            # GPS filter
            df_filter = haversine(df_filter, current_location_lon, current_location_lat, current_radius, gps_areas=selected_gps_label)

        if df_filter.empty == False:
            df_as_html(df_filter)
        else:
            pass

    return df_filter


def gps_info(df_filter):
    df_in = df_filter
    df_filter = df_filter[df_filter['GPS'] != '(None, None)']

    df_filter = df_filter.sort_values('img_path',ascending=True)

    img_path_infobox = []
    for path in df_filter['img_path'].to_list():
        _, tail = os.path.split(path)
        img_path_infobox.append(tail)

    img_path_infobox_all = []
    for path in df_in['img_path'].to_list():
        _, tail = os.path.split(path)
        img_path_infobox_all.append(tail)

    img_path = df_filter['img_path'].to_list()

    date_time =df_filter['date_time'].to_list()
    GPS = eval(str(df_filter['GPS'].to_list()))
    classes_yolo = df_filter['classes_yolo'].to_list()
    classes_ImgNet = df_filter['classes_ImgNet'].to_list()

    return df_filter, img_path, img_path_infobox, img_path_infobox_all, date_time, GPS, classes_yolo, classes_ImgNet


def image_info_sidebox(df_filter):
    photo_obj = Photo()
    df_filter.loc[df_filter.GPS == '(None, None)', "GPS"] = "('None', 'None')"

    df_filter = df_filter.sort_values('img_path',ascending=True)

    img_path_infobox = []
    for path in df_filter['img_path'].to_list():
        _, tail = os.path.split(path)
        tail = os.path.join(photo_obj.upload_folder, tail)
        img_path_infobox.append(tail)

    img_path = df_filter['img_path'].to_list()

    date_time =df_filter['date_time'].to_list()
    GPS = eval(str(df_filter['GPS'].to_list()))
    classes_yolo = df_filter['classes_yolo'].to_list()
    classes_ImgNet = df_filter['classes_ImgNet'].to_list()

    return df_filter, img_path, img_path_infobox, date_time, GPS, classes_yolo, classes_ImgNet


def get_filtered_photoset_to_show(img_path_infobox, img_path_infobox_all, df, selected_gps_label):
    img_path_infobox = [element.replace('\\', '/') for element in img_path_infobox]
    img_path_infobox_all = [element.replace('\\', '/') for element in img_path_infobox_all]

    print('+++++++++++++++++++img_path_infobox+++++++++++++++++')
    print(img_path_infobox)
    print('+++++++++++++++++++img_path_infobox_all+++++++++++++++++')
    print(img_path_infobox_all)
    photo_context = photo_list_classification()
    print('++++++++++++++++photo_context+++++++++++++++')
    photo_list_from_media = [photo.file.name for photo in photo_context['photos']]
    print(len(photo_list_from_media))
    print(photo_list_from_media)

    photos_to_show_all = []
    for photo in photo_context['photos']:
        head, tail = os.path.split(photo.file.name)
        photos_to_show_all.append(tail)

    photos_to_show_display = [os.path.split(photo.file.name)[1] for photo in photo_context['photos'] if  os.path.split(photo.file.name)[1] in img_path_infobox]
    photos_to_show_sorted_indices = [i[0] for i in sorted(enumerate(photos_to_show_display), key=lambda x:x[1])]
    photos_to_show_display = sorted(photos_to_show_display)
    photos_to_show = [photo for photo in photo_context['photos'] if os.path.split(photo.file.name)[1]  in img_path_infobox]
    photos_to_show = [photos_to_show[index] for index in photos_to_show_sorted_indices]
    if selected_gps_label == False:
        print('******************GPS IS OFF ******************')
        photos_to_show_display_viewer = [os.path.split(photo.file.name)[1] for photo in photo_context['photos'] if  os.path.split(photo.file.name)[1] in img_path_infobox_all]
        photos_to_show_sorted_indices_viewer = [i[0] for i in sorted(enumerate(photos_to_show_display_viewer), key=lambda x:x[1])]
        photos_to_show_display_viewer = sorted(photos_to_show_display_viewer)
        photos_to_show_viewer = [photo for photo in photo_context['photos'] if os.path.split(photo.file.name)[1]  in img_path_infobox_all]
        photos_to_show_viewer = [photos_to_show_viewer[index] for index in photos_to_show_sorted_indices_viewer]
    else:
        print('******************GPS IS ON ******************')
        photos_to_show_display_viewer = photos_to_show_display
        photos_to_show_sorted_indices_viewer = photos_to_show_sorted_indices
        photos_to_show_display_viewer = sorted(photos_to_show_display)
        photos_to_show_viewer = photos_to_show


    print('photos_to_show_display')
    print(photos_to_show_display)
    print(len(photos_to_show))
    print(photos_to_show)
    print('photos_to_show_display_viewer')
    print(photos_to_show_display_viewer)
    print(len(photos_to_show_viewer))
    print(photos_to_show_viewer)
    print('************************************************')
    """
    if df.empty:
            photos_to_show_viewer = [photo for photo in photo_context['photos'] if os.path.split(photo.file.name)[1].lower() == "no_data.png"]
    else:
        photos_to_show_viewer = [photo for photo in photos_to_show_viewer if os.path.split(photo.file.name)[1].lower() != "no_data.png"]
    """
    return photos_to_show, photos_to_show_viewer

# Get data choice from website: Yolo, ImageNet, datetime selction, GPS data
def filter_results(request):
    global photos_to_show_viewer
    yolo_flat_list, imageNet_flat_list = dataframe_analysis()

    result_xlsx_path, image_folder = results_dataframe()

    yolo_flat_list = sorted(yolo_flat_list)
    imageNet_flat_list = sorted(imageNet_flat_list)
    yolo_flat_list.insert(0,'all')
    imageNet_flat_list.insert(0,'all')

    # If user triggers an event


    if request.method == 'POST':

        selected_value_yolo = request.POST['drop1']
        selected_value_imageNet = request.POST['drop2']
        selected_logic = request.POST.get('selected_logic_sl', False)
        selected_start_date = request.POST['start']
        selected_end_date = request.POST['end']
        current_location_lat = request.POST['current_location_lat']
        current_location_lon = request.POST['current_location_lon']
        current_radius = request.POST['current_radius']
        current_zoom = request.POST['current_zoom']
        selected_gps_state = request.POST.get('selected_gps_state', False)

        if selected_logic == 'on':
            selected_logic_label = 'OR'
            selected_logic_back = 'checked'
        else:
            selected_logic_label = 'AND'
            selected_logic_back =''

        if selected_gps_state == 'on':
            selected_gps_label = True
            selected_gps_back = 'checked'
        else:
            selected_gps_label = False
            selected_gps_back =''


        df_filter = get_filtered_dataFrame(selected_start_date,
                                                       selected_end_date,
                                                       selected_value_yolo,
                                                       selected_value_imageNet,
                                                       selected_logic_label,
                                                       current_location_lat,
                                                       current_location_lon,
                                                       current_radius,
                                                       selected_gps_label
                                                       )


        df_gps, img_path, img_path_infobox, img_path_infobox_all, date_time, GPS, classes_yolo, classes_ImgNet = gps_info(df_filter)


        photos_to_show, photos_to_show_viewer = get_filtered_photoset_to_show(img_path_infobox, img_path_infobox_all, df_gps, selected_gps_label)
        photos_to_show_viewer_href = [photos.file.url in photos_to_show_viewer]

        lats = [eval(x)[0] for x in df_gps['GPS'].to_list()]
        longs = [eval(x)[1] for x in df_gps['GPS'].to_list()]
        markers_and_infos = zip(lats,
                                longs,
                                img_path,
                                date_time,
                                GPS,
                                classes_yolo,
                                classes_ImgNet,
                                photos_to_show
                                )

        df_filter, img_path_sb, img_path_infobox_sb, date_time_sb, GPS_sb, classes_yolo_sb, classes_ImgNet_sb  = image_info_sidebox(df_filter)

        lats_sb = [eval(x)[0] for x in df_filter['GPS'].to_list()]
        longs_sb = [eval(x)[1] for x in df_filter['GPS'].to_list()]

        markers_and_infos_json = [list(a) for a in zip(lats_sb,
                                longs_sb,
                                [str(im_p) for im_p in img_path_infobox_sb],
                                [str(dt) for dt in date_time_sb],
                                GPS_sb,
                                classes_yolo_sb,
                                classes_ImgNet_sb,
                                )]
        print('das sind die Fotos die in die Thumbviews gehen')
        print(photos_to_show_viewer)
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
            'curr_zoom': current_zoom,
            'photos_class': photos_to_show_viewer,
            'photos_class_href': photos_to_show_viewer_href,
            'selec_gps' : selected_gps_back,
            'markers_and_infos' : markers_and_infos,
            'markers_and_infos_json' : markers_and_infos_json,
            }


    # Initialization of webpage
    else:

        result_xlsx_path, image_folder = results_dataframe()
        df = pd.read_excel(result_xlsx_path)
        df = df.sort_values('img_path',ascending=True)
        df_gps, img_path, img_path_infobox, img_path_infobox_all, date_time, GPS, classes_yolo, classes_ImgNet = gps_info(df)


        photos_to_show, photos_to_show_viewer = get_filtered_photoset_to_show(img_path_infobox, img_path_infobox_all, df_gps, selected_gps_label=False)
        photos_to_show_viewer_href = [photos.file.url in photos_to_show_viewer]
        
        lats = [eval(x)[0] for x in df_gps['GPS'].to_list()]
        longs = [eval(x)[1] for x in df_gps['GPS'].to_list()]
        markers_and_infos = zip(lats,
                                longs,
                                img_path,
                                date_time,
                                GPS,
                                classes_yolo,
                                classes_ImgNet,
                                photos_to_show
                                )

        print('lats')
        print(lats)
        print('img_path')
        print(img_path)
        print('date_time')
        print(date_time)
        print('GPS LIST')
        print(GPS)
        print('classes_yolo')
        print(classes_yolo)
        print('classes_ImgNet')
        print(classes_ImgNet)


        df, img_path_sb, img_path_infobox_sb, date_time_sb, GPS_sb, classes_yolo_sb, classes_ImgNet_sb  = image_info_sidebox(df)

        lats_sb = [eval(x)[0] for x in df['GPS'].to_list()]
        longs_sb = [eval(x)[1] for x in df['GPS'].to_list()]

        markers_and_infos_json = [list(a) for a in zip(lats_sb,
                                longs_sb,
                                [str(im_p) for im_p in img_path_infobox_sb],
                                [str(dt) for dt in date_time_sb],
                                GPS_sb,
                                classes_yolo_sb,
                                classes_ImgNet_sb,
                                )]
        print('markers_and_infos_json')
        print(markers_and_infos_json)

        ctx = {
            'yolo_select': yolo_flat_list,
            'imageNet_select': imageNet_flat_list,
            'selected_yolo': 'all',
            'selected_imageNet': 'all',
            'selec_log': 'checked',
            'selected_start': '01/01/1976',
            'selected_end':  datetime.today().strftime('%m/%d/%Y'),
            'current_loc_lat': '0',
            'current_loc_lon': '0',
            'current_rad': '500',
            'curr_zoom': 2.5,
            'photos_class': photos_to_show_viewer,
            'photos_class_href': photos_to_show_viewer_href,
            'selec_gps' : '',
            'markers_and_infos' : markers_and_infos,
            'markers_and_infos_json' : markers_and_infos_json,
            }

    return render(request, 'f_filter.html', ctx)

def filter_result_table_page(request):
    return render(request, 'g_filter_result.html')

def photo_list_filtered(request):
    global photos_to_show_viewer
    return render(request, "c_photo.html", {'photos': photos_to_show_viewer})


def imageNet_class_object(request):

    answer = request.GET['imageNet_dropdown']
