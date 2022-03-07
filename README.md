# Detection And Classification Of Visible Events In Fusion Reactors Using Deep Learning

This project done by David Serrano for his Final Bachelor Thesis consist in a system capable of detecting and classifying the visible events of Wendelstein 7-X, a experimental fusion reactor build by the IPP (Max Planck Institute for Plasma Physics) which intends to demonstrate the capabilities of fusion power to produce energy. The system has the capability of classifying the visible events between 3 classes: normal events, hot spots (extremely harmful events for the reactor) and anomalies (neither normal events nor hot spots such as falling debris or pellets). The report of this thesis is available in the following link. In this repository all the implemented code is stored in jupyter notebooks to have a pleasant follow-up. However, in this README, the functionality of each notebook is overviewed.

<a href="TFG_DavidSerrano.pdf" target="_blank">PDF Report</a>

<a href="https://upcommons.upc.edu/handle/2117/356904" target="_blank">UPCommons</a>

#### 1_Tracker.ipynb

This file analyzes the video sequences, finds candidates of HotSpots, track them and links them in the neighboring frames. It also permits labeling them manually between NoHotSpot, HotSpot and Anomaly to later on implement a classifier to do it automatically.



