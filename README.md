# Breast Cancer Detection

This repository serves a documentation for the work done on detecting and classifying cancer in breast mammograms during the Spring 2019-2020 term. The models, datasets, and experiment outline are the result of Layal Al Khuja's research work, and the implementation is coded by Christophe Karam. The repository contains Jupyter notebooks containing the code necessary to run each bundle of experiments (along with the proper environment setup and data processing). The repository also contains the datasets used, and instructions on how to run everything. The following are thoughts on the experimentation, the models, and the datasets, which can be used as a basis for future research.

## Discussion

### Datasets

There are 4 target datasets: MIAS, InBreast, DDSM, and BCDR-F03. 
The MIAS and InBreast dataset were obtained and processed, instructions on how to download are in the next section.
The BCDR-F03 dataset is obtainable through [here](https://bcdr.ceta-ciemat.es). However, an account has to be set-up and a non-discolsure agreement has to be filled out and sent to the authors. This was done but the authors never responded, so the dataset was not obtained.

The DDSM dataset is publicly available [here](http://www.eng.usf.edu/cvprg/Mammography/Database.html). The raw dataset is over 200GB in size, and is not in a usable format. The dataset can be downloaded in parts, and converted to a usable format using [this repo](https://github.com/nicholaslocascio/ljpeg-ddsm), but this repository's code depend on another repository which presented compatibility issues when tested, but maybe there are ways around this. Since the models also presented problems, processing this dataset was abandoned eventually.

### Models

Besides the 4 datasets, of which only 2 were obtained, we have 6 models. Layal provides repository links to 3 of those 6 models. The last model involves large scale training and model selection, and has no reference code, so will definitely not be tested. The remaining two models without reference code are fairly simple and have been implemented in Keras according to the papers, so they have been recreated from scratch.

The model from article 1 uses the BCDR dataset. Several challenges present themselves in this case: the original model cannot be properly tested due to the unavailability of the dataset, as well as the deprecation of most libraries used in the provided repository, `pylearn2` and `shapely`, and even `python 2.7`. The compatibility issues rendered this task not doable.

The model in article 4 is an intruder here, being a object-detection model and not a classifier. The code requires [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) to work which is deprecated and also dependent itself on a deprecated version of Caffe, and it only provides instructions on how to run the pretrained model for detection, but not on how to train. The compilation of the dependencies failed, but at least it seems the ROI data for the INbreast dataset are provided in the repository. Possible solutions can be:
1. Find a way to compile the dependencies, shouldn't be too difficult, and read the old Caffe documentation on how to train a model. The data is in the repository along with a model configuration file with a `.prototxt` extension which might be all that is needed. In case additional information is missing, training won't be possible.
2. [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) states that it is deprecated along with its dependencies, and suggests trying out Detectron (newer Caffe, sort of). Detectron itself says that even it is deprecated, and suggests trying out Detectron2. This can be a possibility, but in this case we would have ditched all the older frameworks for their much newer versions, and the only information we have is that single `.prototxt`  file with the model structure.
3. Last possible alternative is to ditch this entire solution and train another well-tested repository, since nothing novel is being done here and this work is fairly old.

The model in article 2 is very well documented, and does not use any outdated libraries, and even provides implementations in both Tensorflow and PyTorch. However the problem here is the dataset structure. It is possible to run the provided pretrained models to predict on new images, but the transfer learning part is not as simple because the model uses 4 input images, two views per side (left or right breast), with the additional possibility of adding heatmaps to the input, and it involves a tremendous amount of image processing and conversions between different formats.

## Instructions

The main `BreastCancerDetection.ipynb` notebook can be run in Google Colab, and is divided into sections for easier navigation.

The dataset folder `breast_cancer_data.zip` is provided, and its path only needs to be replaced in the `Environment Setup` section of the notebook. If not, the `MIAS Dataset Downloading and Preprocessing` and the `INbreast Dataset Downloading and Preprocessing` contain links to download and process the original datasets. The rest of the notebook provides the necessary code to create and train the models described in article 3 of the experiment plan, `Deep Convolutional Neural Networks for breast cancer screening`. In the same fashion, the models for article 4 can just as easily be implemented using Keras. More importantly, the dataset preprocessing can be tweaked to obtain different bounding boxes for cancerous areas in the breast mammogram, since the classifiers are trained on a cropped region of interest, and not the entire image.


 
