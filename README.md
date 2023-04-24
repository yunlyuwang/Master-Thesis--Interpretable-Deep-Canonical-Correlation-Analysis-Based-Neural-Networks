# Master-Thesis--Interpretable-Deep-Canonical-Correlation-Analysis-Based-Neural-Networks
## Introduction
It focuses on the topic of Multi-view Data Analysis, using machine learning to analyse image data. 
Specifically, CCA family of algorithms to find the optimal projection with the maximal correlation among the multi-view data. 
Moreover using the deep network visualization techniques to analyse the overfitting problem in this experienment.

<img src="/Images/framework.jpg" width="950">

## Multi-view data
Simplely, multi-view data can be understand as the data got into different process, for example the same meaning words in different languages or the same signal captured by different devices.
<img src="/Images/multiview-2.drawio.jpg" width="800">

## Canonical Correlation Analysis (CCA) family of algorithms
- CCA 
CCA can find the linear projections that the new represetations are maximally correlated.
<img src="/Images/CCA-Page-1.drawio.jpg" width="400">

- DCCA
DCCA makes use of the deep neural  network to process the multi views first, then using CCA to get the maximally correlated projections.
<img src="/Images/DCCA.jpg" width="600">

- DCCAE
Based DCCA, adding the decoder part to reconstruct the original data is the key of DCCAE
<img src="/Images/DCCAE.jpg" width="600">

## Visualization of neural network
- Saliency Map

An example of a classification model:

<img src="/Images/dog_saliency_map.png" width="500">

Using the gradients that generate ouput backpropogate though the network can get the saliency map for the neural network. 
It can help to evaluate performance of the model.
<img src="/Images/sm.jpg" width="800">

- SmoothGrad
<img src="/Images/smooth.jpg" width="800">

- Gradient-weighted Class Activation Mapping(GradCAM) 
<img src="/Images/gradcam.jpg" width="800">


## Evaluation of proposed model
Shown by the visualation techniques.
<img src="/Images/c2b-157.jpg" width="1000">
It shows that when it runs too many epochs which is more possibly overfitting it could not find the correlation by the semantical pixels.
