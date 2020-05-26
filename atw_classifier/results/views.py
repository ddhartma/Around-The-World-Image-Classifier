from django.shortcuts import render
from django.conf import settings

from photos.views import photo_list_classification
from classification.views import check_folderPaths

import requests
import os
import subprocess
from shutil import copyfile
from datetime import datetime, date, timedelta

from django.http import HttpResponse
import pandas as pd
from collections import Counter

import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)

from django.contrib.auth.decorators import login_required

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

    print('++++++++++++++++++++++++++++++++result_html_path+++++++++++++++++++++++')
    print(result_html_path)

    # Build paths inside the project like this: os.path.join(BASE_DIR, ...)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print('*****************Destination path+++++++++++++++++')
    dst_path = os.path.join(BASE_DIR, 'templates', 'table.html')
    print('dst_path')
    print(dst_path)
    copyfile(result_html_path, dst_path)

    return result_xlsx_path, result_imgPerDay_xlsx_path

def dataframe_analysis():
    result_xlsx_path, result_imgPerDay_xlsx_path = results_dataframe()

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

    # images total
    num_img_total = str(df.shape[0])

    # yolo classifications
    yolo_list = [eval(x) for x in df['classes_yolo'].to_list()]
    num_img_classified_yolo = len([element for element in yolo_list if element != []])


    # ImageNet classifications
    imageNet_list = [eval(x) for x in df['classes_ImgNet'].to_list()]
    print('imageNet_list')
    print(imageNet_list)
    num_img_classified_imageNet = len([element for element in imageNet_list if element != []])
    print('num_img_classified_imageNet')
    print(num_img_classified_imageNet)

    # GPS
    gps_list = eval(str(df['GPS'].to_list()))
    num_img_gps = len([x for x in gps_list if x !='(None, None)'])

    # DateTime Range - set by user
    dateTime_range_early = df_count['date'].iloc[0]
    dateTime_range_late = df_count['date'].iloc[-1]

    # DateTime Range found from image analysis
    df_count_range = df_count[df_count['date_count'] != 0]
    img_early_date = df_count_range['date'].iloc[0]
    img_late_date = df_count_range['date'].iloc[-1]

    # Missing days (days without images)
    num_miss_days = len(df_count) - len(df_count_range)
    print('num_miss_days')
    print(num_miss_days)

    # of images with people
    print(yolo_list)
    num_img_human_list = []

    for element in yolo_list:
        if 'person' in element:
            num_img_human_list.append(True)

    num_img_human = len(num_img_human_list)

    # of images w/o people
    num_img_wo_human = len(yolo_list) - num_img_human

    # mean number of humans for images with humans

    flat_list = [item for sublist in yolo_list for item in sublist]
    yolo_classes_to_int_dict =  Counter(flat_list)
    print(yolo_classes_to_int_dict)

    mean_human_num_per_img = yolo_classes_to_int_dict['person'] / len(num_img_human_list)

    # total number of human beings identified
    num_total_humans = yolo_classes_to_int_dict['person']

    return num_img_total, num_img_classified_yolo, num_img_classified_imageNet, num_img_gps, dateTime_range_early, dateTime_range_late, img_early_date, img_late_date, num_miss_days, num_img_human, num_img_wo_human, mean_human_num_per_img, num_total_humans

def layout(title_fig, title_size, title_xaxis_size, title_xaxis_ticks_size, title_yaxis_size, title_yaxis_ticks_size, legend_font_size, x_title, y_title, legend_text_from_dict):
    lay_out = {'title' : title_fig, "title_font": {"size": title_size},
             'xaxis' : {'title' : x_title, "title_font": {"size": title_xaxis_size}, "tickfont": {"size": title_xaxis_ticks_size}, 'zeroline' : True},
             'yaxis' : {'title' : y_title, "title_font": {"size": title_yaxis_size}, "tickfont": {"size": title_yaxis_ticks_size}, 'zeroline' : True},
             'legend' : {'title' : legend_text_from_dict, "title_font": {"size": legend_font_size} , "font": {"size": legend_font_size}},
             'autosize' : True,
             'margin' : {'l':100, 'r': 100, 't': 100, 'b': 100},

                         }
    return lay_out


def plot_data():
    result_xlsx_path, result_imgPerDay_xlsx_path = results_dataframe()

    df = pd.read_excel(result_xlsx_path)
    df_count = pd.read_excel(result_imgPerDay_xlsx_path)

    #######################
    # Yolo classification #
    #######################
    yolo_list = [eval(x) for x in df['classes_yolo'].to_list()]

    flat_list_yolo = [item for sublist in yolo_list for item in sublist]
    yolo_classes_to_int_dict =  Counter(flat_list_yolo)
    print(yolo_classes_to_int_dict)


    # bar plot: number of images as function of yolo class
    trace_yolo = go.Bar(x= list(yolo_classes_to_int_dict.keys()),
                   y= list(yolo_classes_to_int_dict.values()),
                   )

    title_fig = 'bar plot: # of images / yolo class'
    title_size = 24
    title_xaxis_size = 20
    title_xaxis_ticks_size = 16
    title_yaxis_size = 20
    title_yaxis_ticks_size = 16
    legend_font_size = 20
    x_title = 'Yolo classes'
    y_title = 'counts'
    legend_text_from_dict=''



    layout_yolo = layout(title_fig, title_size, title_xaxis_size, title_xaxis_ticks_size, title_yaxis_size, title_yaxis_ticks_size, legend_font_size, x_title, y_title, legend_text_from_dict)

    data_yolo = [trace_yolo]
    fig_yolo = go.Figure(data=data_yolo,  layout = layout_yolo)


    ##########################
    # ImageNet classifcation #
    ##########################
    imageNet_list = [eval(x) for x in df['classes_ImgNet'].to_list()]

    flat_list_imageNet = [item for sublist in imageNet_list for item in sublist]
    imageNet_classes_to_int_dict =  Counter(flat_list_imageNet)
    print(imageNet_classes_to_int_dict)


    # bar plot: number of images as function of yolo class
    trace_imageNet = go.Bar(x= list(imageNet_classes_to_int_dict.keys()),
                   y= list(imageNet_classes_to_int_dict.values()),
                   )

    title_fig = 'bar plot: # of images / ImageNet class'
    x_title = 'ImageNet classes'
    y_title = 'counts'
    legend_text_from_dict=''

    layout_imageNet = layout(title_fig, title_size, title_xaxis_size, title_xaxis_ticks_size, title_yaxis_size, title_yaxis_ticks_size, legend_font_size, x_title, y_title, legend_text_from_dict)

    data_imageNet = [trace_imageNet]
    fig_imageNet = go.Figure(data=data_imageNet,  layout = layout_imageNet)


    ##################
    # Images per day #
    ##################

    # bar plot: number of images as function of yolo class
    trace_counts_per_day = go.Bar(x= df_count['date'].to_list(),
                   y= df_count['date_count'].to_list())


    title_fig = 'bar plot: # of images / day'
    x_title = 'date'
    y_title = 'counts'
    legend_text_from_dict=''

    layout_counts_per_day = layout(title_fig, title_size, title_xaxis_size, title_xaxis_ticks_size, title_yaxis_size, title_yaxis_ticks_size, legend_font_size, x_title, y_title, legend_text_from_dict)

    data_counts_per_day = [trace_counts_per_day]
    fig_counts_per_day = go.Figure(data=data_counts_per_day,  layout = layout_counts_per_day)


    #########################
    # Export graphs as HTML #
    #########################

    # Build paths inside the project like this: os.path.join(BASE_DIR, ...)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dst_path_yolo = os.path.join(BASE_DIR, 'templates', 'bar_plot_yolo_counts.html')
    dst_path_imageNet = os.path.join(BASE_DIR, 'templates', 'bar_plot_imageNet_counts.html')
    dst_path_counts_per_day = os.path.join(BASE_DIR, 'templates', 'bar_plot_counts_per_day.html')
    fig_yolo.write_html(dst_path_yolo)
    fig_imageNet.write_html(dst_path_imageNet)
    fig_counts_per_day.write_html(dst_path_counts_per_day)

@login_required
def results(request):
    _ ,_ = results_dataframe()
    return render(request, 'd_results.html')


@login_required
def analysis(request):
    num_img_total, num_img_classified_yolo, num_img_classified_imageNet, num_img_gps, dateTime_range_early, dateTime_range_late, img_early_date, img_late_date, num_miss_days, num_img_human, num_img_wo_human, mean_human_num_per_img, num_total_humans = dataframe_analysis()

    plot_data()

    return render(request, 'e_analysis.html', {'num_img_total': num_img_total,
                                               'num_img_classified_yolo': num_img_classified_yolo,
                                               'num_img_classified_imageNet': num_img_classified_imageNet,
                                               'num_img_gps': num_img_gps,
                                               'dateTime_range_early': dateTime_range_early,
                                               'dateTime_range_late': dateTime_range_late,
                                               'img_early_date': img_early_date,
                                               'img_late_date': img_late_date,
                                               'num_miss_days': num_miss_days,
                                               'num_img_human': num_img_human,
                                               'num_img_wo_human': num_img_wo_human,
                                               'mean_human_num_per_img': mean_human_num_per_img,
                                               'num_total_humans' : num_total_humans,
                                               })
