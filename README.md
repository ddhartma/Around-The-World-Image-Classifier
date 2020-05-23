[//]: # (Image References)

[image1]: dog_pred_example.png "Sample Output"

# Around The World Photo Classifier

This is a Django webserver application. It incorporates Jupyter notebook operation and web page visualization of a deep learning - image classification topic. Via Jupyter notebooks the deep learning events and data exctration (image classification, GPS data, datetime) are processed, evaluated and documentated. Image uploads, data analysis and classification results are organized and shown via web pages. Data analysis visualization is supported by Plotly plots, tables and image bags.


All Deep Learning classification events are placed inside the Jupyter notebook **around_the_world_classifier.ipynb**. It has four purposes:
1. It extracts datetimes (if available)
2. It extracts GPS data (if available)
3. It provides object classification via yolo3v, ImageNet and personalized CNNs
4. It filters, sorts and rearranges images based on the extraced data

Datetimes and GPS data are extracted from the image meta data using the pillow library.

The photo classification is realized in three different approaches:
1. by using a yolov3 object detection algorithm. Here, I am using yolov3 pretrained wheights. Deep Learning Inference with own images enables a detection of up to 80 different classes within one image. A Boundary box with a class description is provided and stored in a separate folder **.../path_to_your_image_folder_name_** _yolo_class_  as well as stored in an html table (**your_image_folder_name.html**).
2. by using a CNN based pretrained model from Torchvision via Transfer Learning. As a standard VGG16 is chosen. However, you can replace VGG16 by any ozjer torchvision model. VGG16 is using the whole ImageNet classification system, The total number of classes is 1000. The file **data/imagenet_classes.txt** provides a dictionary of all one 1000 classes.
3. by using an own CNN based architecture with a layer combination of three times 'Conv-ReLU-MaxPool', deeply enough for sufficient feature extraction and an appropriate image size/feature reduction. The goal of this CNN is to filter peronalized images, e.g. to identify images of yourself. However, for this classification step you have to provide a dataset of at least 300 personolized images. This CNN is not pretrained. You have to train it separately. For further information, follow the instruction in the Jupyter Notebook.



## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- To run this script you will need to use a Terminal (Mac OS) or Command Line Interface (Git Bash on Windows).
- If you are unfamiliar with Command Line check the free [Shell Workshop](https://www.udacity.com/course/shell-workshop--ud206) lesson at Udacity.

### Installation Part A - Python and Environment

- To run this script you will need to use a Terminal (Mac OS) or Command Line Interface (e.g. Git Bash on Windows [git](https://git-scm.com/)), if you don't have it already.
- If you are unfamiliar with Command Line coding check the free [Shell Workshop](https://www.udacity.com/course/shell-workshop--ud206) lesson at Udacity.


Besides Python you need to install several scientific Python libraries that are necessary for this project, in particular NumPy, Matplotlib, Pandas, Jupyter Notebook, Opencv, PyTorch and Torchvision. Since I have many projects with different library requirements, I prefer to use isolated environments.

I provide two approaches for Python + environment installation:

#### First approach - via yml-file:

- Install a conda environment from the provided environment.yml file (I created this file via `conda env export > environment.yml`).

  ```
  $ conda env create -f atw_photo_class.yml
  ```


#### Second approach - manually:

- **Phython istallation:** Of course, you obviously need Python. Python 3 is already preinstalled on many systems nowadays. You can check which version you have by typing the following command (you may need to replace python3 with python):

  ```
  $ python3 --version  # for Python 3
  ```
  Any Python 3 version should be fine, preferably 3.5 or above. If you don't have Python 3, you can just download it from [python.org](https://www.python.org/downloads/).

  These are the commands you need to type in a terminal if you want to use pip to install the required libraries. If possible use conda for installation.

  First you need to make sure you have the latest version of conda installed:

  ```
  $ conda update conda
  ```

  Next, create an isolated environment. This is recommended as it makes it possible to have a different environment for each project (e.g. one for this project), with potentially very different libraries, and different versions:

  ```
  $ conda create -n atw_photo_class python=3.6
  ```

  This creates a new isolated Python environment based on Python 3. If you wish a higher Phyton version (e.g. Python 3.7), feel free to change it.

  Now you must activate this environment. You will need to run this command every time you want to use this environment.

  On Windows:
  ```
  $ source activate atw_photo_class
  ```
  On Mac:
  ```
  $ conda activate atw_photo_class
  ```

- **Important libraries:**
  - Install NumPy, Pandas, Matplotlib
    ```
    $ conda install numpy
    $ conda install pandas
    $ python -m pip install -U pip
    $ python -m pip install -U matplotlib
    ```

  - Install [PyTorch](https://pytorch.org/?utm_source=Google&utm_medium=PaidSearch&utm_campaign=%2A%2ALP+-+TM+-+General+-+HV+-+GER&utm_adgroup=Conda+Install+PyTorch&utm_keyword=%2Bconda%20%2Binstall%20%2Bpytorch&utm_offering=AI&utm_Product=PyTorch&gclid=CjwKCAjw3-bzBRBhEiwAgnnLCszXVKwBc_0Rjx6qpPPqiq7NwCwm0nEqIXOsNrrqJ4lZ1FMCVF4nhxoCeZUQAvD_BwE) for your system. Use the provided conda or pip command in your terminal for your specific system (e.g. Mac or Windows, CPU or GPU).

  - Install gmaps via

    ```
    $ conda install -c conda-forge gmaps
    ```

  - Install opencv-python via

    ```
    $ conda install -c conda-forge opencv
    ```
  - Install pillow
    ```
    $ pip install Pillow==6.0.0
    ```
    Important Note: Use a version of Pillow <7.0 if you want to analyse jpg file formats. Pillow >=7.0 leads to error messages.

### Installation Part B - Clone the Repo, Download yolov3 weights, provide your dataset

- Clone this repository by opening a terminal and typing the following commands:

  ```
  $ cd to your development directory. Eventually create a new one via:
  $ (mkdir folder_name)
  $ (cd folder_name)
  $ git clone https://github.com/ddhartma/tbd.git
  $ cd tbt
  ```

- Download the [yolov3 weights](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Place this file in the repo-folder, at location `path/to/data/config`.  This file contains the pretrained wheights for a yolov3 image detection with 80 classes. More details can be found [here](https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98).

- Place your dataset (stored in one folder) in the tbt folder.


Great! You're all set, you just need to start Jupyter now.

## Running the tests

The following files were used for this project:

- around_the_world_classifier.ipynb.ipynb
- folder **data**. This folder contains the **config** folder containing yolov3 files.
- folder **utils** (important for yolov3 pretrained learning).

Open a terminal window and navigate to the project folder. Open the notebook and follow the instructions.
```
jupyter notebook around_the_world_classifier.ipynb
```

- Follow the instructions in the notebook.

__NOTE:__ In the notebook, you will need to train CNNs in PyTorch.  If your CNN is taking too long to train, feel free to pursue one of the options under the section __Accelerating the Training Process__ below.



## (Optionally) Accelerating the Training Process

If your code is taking too long to run, you will need to either reduce the complexity of your chosen CNN architecture or switch to running your code on a GPU.  If you'd like to use a GPU, you can spin up an instance of your own:

#### Amazon Web Services

You can use Amazon Web Services to launch an [EC2 GPU instance](https://aws.amazon.com/de/ec2/). However, this service is not for free.

## Acknowledgments
* Please check out great Udacity Nanodegree programs, e.g. [Deep Learning](https://www.udacity.com/course/deep-learning-nanodegree--nd101)
