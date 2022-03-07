# Detection And Classification Of Visible Events In Fusion Reactors Using Deep Learning

This project done by David Serrano for his Final Bachelor Thesis consist in a system capable of detecting and classifying the visible events of Wendelstein 7-X, a experimental fusion reactor build by the IPP (Max Planck Institute for Plasma Physics) which intends to demonstrate the capabilities of fusion power to produce energy. The system has the capability of classifying the visible events between 3 classes: normal events, hot spots (extremely harmful events for the reactor) and anomalies (neither normal events nor hot spots such as falling debris or pellets). The report of this thesis is available in the following link. In this repository all the implemented code is stored in jupyter notebooks to have a pleasant follow-up. However, in this README, the functionality of each notebook is overviewed.

[PDF report]()
<a href="https://upcommons.upc.edu/handle/2117/356904" target="_blank">UPCommons</a>

### 1_Tracker.ipynb


The system first applies a number of techniques (TopHat, Otsu Binarization...) to obtain all the abnormal visible events. Then using a ResNet50, 640 features are obtained for every candidate and sequentially classified between the 3 classes using a Machine Learning classifier (SVM and XGBoost).

https://user-images.githubusercontent.com/61697910/132984390-1192729a-2be8-4bd0-9388-825d314e36d5.mp4

