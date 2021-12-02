# Clustering and anomaly detection on categorical data

## Requirements

The environment works with Python 3.8.11, the conda version is 4.9.2

Keras and tensorflow are only required for the categorical embeddings. If you only want to use the anomaly detection package, then ignore those two modules.

The overall code is relatively simple and does not use any complex functions, so I assume any new versions of the following packages should work for a long time. Just in case, I'll provide you with the versions I'm using:

* keras==2.6.0

* matplotlib==3.4.2

* numpy==1.21.1

* pandas==1.3.1

* scipy==1.7.0

* scikit-learn==0.24.2

* tensorflow==2.6.0

* tensorflow-probability==0.14.0

If you are using conda, you can run

```
conda env create -f environmentFinal.yml
```

in the project folder to install exactly all the packages (and versions) I have used.

Bear in mind that I was unable to make tensorflow work with gpu, so any algorithms or models that make use of it will work slower, as they will use the cpu instead (if the user can make it work, it should be much faster).

## Introduction

Both clustering and anomaly detection are relevant subjects in machine learning. However, not many algorithms are able to properly handle categorical data, even though most real datasets tend to contain a significant ratio of categorical attributes.

The employed dataset is a modified version of the uci mushroom dataset [4], where missing values have been inputted in RStudio making use of the mice package. All data has also been one-hot encoded. I chose this dataset beacuse of the large amount of measurements, dimensionality, and the fact that all attributes are categorical.

These tasks become more complex and harder to understand when data dimensionality is high. In this repository I have attempted to board these tasks (with a special focus on anomaly detection), guiding the user through each step. I have developed a custom class, named find_outliers (in AnomalyDetection.py), to speed up the process of outlier detection. All four algorithms are from the sklearn library [5] and usually provide different results (see CaseOfUseAnomalyDetection.ipynb for more information).

Most clustering algorithms are also from the sklearn library, even though I have tested two R clustering algorithms (named ROCK and HDBSCAN), as well as a custom python implementation of quantum clustering made by Carlos Hernani [3].

I have also attempted to convert the categorical data from OHE vectors into dense embeddings, which has improved the results and interpretability (see CaseOfUseCategoricalEmbeddings.ipynb).

## Rough guide

Since clustering and anomaly detection processes heavily depend on the type of data being studied [1], I can only offer a general guide without going into too much detail. There are, however, a list of steps that are generally followed:

1) Data cleaning and variable selection: Without a proper selection of relevant variables all following steps are, very likely, going to fail. Some clustering algorithms try to find a subset of variables in which clustering is easier, but this initial step is still necessary.

2) Clustering: Depending on the structure of the data, some algorithms may work better than others. Most tested algorithms work well and provide mostly pure clusters. (the user can check the results in CategoricalClustering.ipynb).

3) Anomaly detection: Even though the anomaly detection algorithms can be applied over the entire dataset, it is recommended to study each cluster separately [1], so that the models can learn the particularities of each one of them (see CaseOfUseAnomalyDetection.ipynb). Even though the IsolationForest is fast and efficient, the Local Outlier Factor seems to provide good results when dimensionality is high [2]. A comparison of all the models can also be seen in [5].

4) The usage of categorical embeddings (see CaseOfUseCategoricalEmbeddings.ipynb) for this last step may improve the results, as all of these models were made to work with purely numerical attributes. Employing One-Hote-Encoded vectors will still work, but may not provide the best results.

## References

[1] Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. ACM computing surveys (CSUR), 41(3), 1-58. (url: https://dl.acm.org/doi/abs/10.1145/1541880.1541882)

[2] Zimek, A., Schubert, E., & Kriegel, H. P. (2012). A survey on unsupervised outlier detection in high‐dimensional numerical data. Statistical Analysis and Data Mining: The ASA Data Science Journal, 5(5), 363-387. (url: https://onlinelibrary.wiley.com/doi/full/10.1002/sam.11161)

[3] Quantum clustering python implementation: https://github.com/carlos-hernani/QlusterPy

[4] Testing dataset(original): https://archive.ics.uci.edu/ml/datasets/mushroom

[5] Anomaly detection comparison: https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html#sphx-glr-auto-examples-miscellaneous-plot-anomaly-comparison-py

[6] Categorical embeddings 1: https://towardsdatascience.com/categorical-embedding-and-transfer-learning-dd3c4af6345d

[7] Categorical embeddings 2: https://towardsdatascience.com/deep-embeddings-for-categorical-variables-cat2vec-b05c8ab63ac0