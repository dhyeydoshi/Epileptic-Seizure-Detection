# Epileptic-Seizure-Detection

The epileptic seizure detection project aims to classify EEG signals into two classes: normal and epileptic. EEG signals were collected from patients with epilepsy and healthy individuals. Machine learning models were used to classify these signals. 

The dataset used for this project contains 5,000 EEG signal recordings, each of which is 23.6 seconds long. The data was preprocessed and features were extracted using various techniques, including statistical features, spectral analysis, and wavelet transforms. 

Several machine learning models were trained and tested on this dataset, including K-Nearest Neighbors, Support Vector Machines, Random Forest, Naive Bayes, Logistic Regression, and Convolutional Neural Networks. The best-performing model was the CNN, achieving an accuracy of 98.5%.

The code for this project is written in Python using various libraries such as Pandas, NumPy, Scikit-learn, TensorFlow, and Keras. A Django web application was built to upload and preprocess new EEG data, and predict whether they are normal or epileptic.

Overall, this project demonstrates the potential of machine learning models in accurately classifying EEG signals for epilepsy diagnosis.
