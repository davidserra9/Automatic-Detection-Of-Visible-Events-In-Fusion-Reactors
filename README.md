# Detection And Classification Of Visible Events In Fusion Reactors Using Deep Learning

This project done by David Serrano for his Final Bachelor Thesis consist in a system capable of detecting and classifying the visible events of Wendelstein 7-X, a experimental fusion reactor build by the IPP (Max Planck Institute for Plasma Physics) which intends to demonstrate the capabilities of fusion power to produce energy. The system has the capability of classifying the visible events between 3 classes: normal events, hot spots (extremely harmful events for the reactor) and anomalies (neither normal events nor hot spots such as falling debris or pellets). The report of this thesis is available in the following link. In this repository all the implemented code is stored in jupyter notebooks to have a pleasant follow-up. However, in this README, the functionality of each notebook is overviewed.

<a href="TFG_DavidSerrano.pdf" target="_blank">PDF Report</a>

<a href="https://upcommons.upc.edu/handle/2117/356904" target="_blank">UPCommons</a>

#### 1_Tracker.ipynb

This file analyzes the video sequences, finds candidates of HotSpots, track them and links them in the neighboring frames. It also permits labeling them manually between NoHotSpot, HotSpot and Anomaly to later on implement a classifier to do it automatically.

#### 2_DataAnalysis.ipynb

This file analyzes how many sequences have been labeled, how many detections and tracks we have found (and their labels) and how the detections Bounding Boxes and tracks are.

#### 3_DataPreprocessing.ipynb

This file creates several important functions to preprocess both the detections and images from the sequences.

#### 4_CustomDataLoader.ipynb

This file implements the Dataset and Dataloaders classes, as well as diving the dataset into training and test and balancing the training dataset.

Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

#### 5_NNwithSoftmax.ipynb

This file uses the manually labeled detections to classify them automatically using Transfer Learning and Fine Tunning.

#### 6_FeatureExtraction.ipynb

This file extracts the features of the fully trained ResNet50.

#### 7_SVM.ipynb

This file gets the stored features to classify the tracks using SVM. All the resampling techniques explained in the report are used.

#### 8_XGBOOST.ipynb

This file gets the stored features to classify the tracks using XGBoost. All the resampling techniques explained in the report are used.

#### 9_System-wideTests.ipynb

This file tests the entire system with unseen sequences.

#### 10_LabelingTool.ipynb

This file takes the prediction of the system-wide tests of unseen sequences and displays 4 images of all the tracks to have the possibility of reviewing the track predicitons.

#### 11_OnlineClassification.ipynb

This file analysis a track using the online system.

#### 12_HDF5toVideo.ipynb

This file transforms the hdf5 file of a sequence into a .avi sequence to be able to display the stream
