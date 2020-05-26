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

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit
- You need a Command Line Interface (CLI). If you are a Window user install [git](https://git-scm.com/)). Under Mac OS use the pre-installed Terminal.
- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```
The project installation is divided into two parts: Part A describes the cloning of the project and the installation of the project environment. Part B describes Yolo weight settings for Transfer Learning, and the implementation of specific Django webserver settings.

### Project installation Part A
- Create a new project folder, e.g. `atw`
- Open Git Bash (Terminal, respectively)
- Change Directory to the newly created folder, e.g. `cd atw`
- Clone the Github Project. Inside Git Bash (Terminal) write:
```
$ git clone https://github.com/ddhartma/Around-The-World-Image-Classifier.git
```

- Change Directory
```
$ cd Around-The-World-Image-Classifier
$ cd atw_classifier
```

- Create a new Python environment via the provided yml file. Inside Git Bash (Terminal) write:
```
$ conda env create -f MTP_LSTM.yml
```

- Check the environment installation via
```
$ conda env list
```

- Activate the installed MTP_LSTM environment via
```
$ conda activate MTP_LSTM (Mac OS)
$ source activate MTP_LSTM (in case of trouble under Windows)
```

### Project installation Part B
- Download the [yolov3 weights](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Place this file in the repo-folder, at location `path/to/data/config`.  This file contains the pretrained wheights for a yolov3 image detection with 80 classes. More details can be found [here](https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98).











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

- around_the_world_classifier.ipynb
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
