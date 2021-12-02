#Class library
#import attr

# Libraries import
import os#Create libraries
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import pickle as pkl
import warnings

#Sklearn libraries
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy import stats
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler

#@attr.s
class find_outliers:
    '''
    Function to find outliers in a given dataset, without specifying clusters in the data.
    Inputs:
    - data: Pre-processed pandas dataframe or numpy array, with rows as points and columns as features.
    - modelPath: String indicating directory where models will be saved. In outlier detection it is a good
      practice to separate models into different clusters, as each cluster will have a different structure
      and outliers may behave differently in other clusters
    - metric: (String) Initial similarity type to calculate the similarity matrix.
    - decomposition: String indicating type of decomposition to apply to dataset, named PCA, SVD or None (using original dataset).
    - dec_components: Integer indicating the number of components in case a PCA or SVD is performed.
    - contamination: Integer or float indicating the estimate of contamination of the dataset 
      (Integer => Top k outliers according to the scores provided by the model, Float (between 0 and 0.5)=> Top outliers 
      making use of the algorithm's parameters to select them). If None is selected, the boxplot method will be employed
      to select outlier candidates, making use of the scores provided by the model. 
    - random_state: Integer indicating a random state to be used in the algorithms, 273 by default.
    Methods:
    All methods output the following objects:
    - Labels for entire dataset (1 = Inlier, -1 = Outlier).
    For a more in-depth comparison between models, check:
    https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html#sphx-glr-auto-examples-miscellaneous-plot-anomaly-comparison-py
    '''
    def __init__(self, data = None, modelPath: str ='ODModels', metric: str = 'euclidean', 
                 decomposition: str = None, dec_components: int = 5,
                 contamination = None, random_state: int = 273):
        #Check current workspace and create directory to save ODmodels
        path = os.getcwd() + '/' + modelPath
        #Check if directory already exists, otherwise create it
        if(os.path.isdir(path)):
            print(f"Saving models at: {path}")
        else:
            try:
                os.makedirs(path)
            except OSError:
                warnings.warn(f"Creation of the directory {path} failed")
            else:
                print(f"Successfully created the directory {path}")
        #Save directory path so that it can be accessed by the models
        self.savepath = path
        #Save data and other key variables into class
        self.data = data
        if(isinstance(data, pd.core.frame.DataFrame)):
            self.X = np.array(data)
        elif(isinstance(data, np.ndarray)):
            self.X = data
        else:
            raise('data must be either a pandas dataframe or a numpy array')
        self.random_state = random_state
        self.decomposition = decomposition
        if(self.decomposition == 'PCA'):
            self.PCA = PCA(n_components = dec_components, random_state=self.random_state)
            self.X = self.PCA.fit_transform(self.X)
        elif(self.decomposition == 'SVD'):
            self.SVD = TruncatedSVD(n_components = dec_components, n_iter=10, random_state=self.random_state)
            self.X = self.SVD.fit_transform(self.X)
        self.metric = metric
        #Calculate metric so that it only has to be calculated once
        self.similarity = pairwise_distances(self.X, metric = self.metric)
        #Handle possible values of contamination
        if(contamination == 'auto'):
            contamination = None
        self.contamination = contamination
        if(not isinstance(self.contamination, (float, int)) and self.contamination != None):
            raise ValueError('contamination must be of type integer, float (between 0 and 0.5) or None.')

    def get_similarity_matrix(self):
        '''
        Method that returns the similarity matrix
        '''
        return(self.similarity)
    
    def get_processed_data(self):
        '''
        Method that returns the processed data, which has been converted into a numpy array and transformed with a PCA or SVD
        in case it was specified when instantiating the class. Remember that PCA and SVD should preferably be employed when 
        the entire dataset is being processed.
        '''
        return(self.X)
    
    def elliptic_envelope(self, store_precision=True, assume_centered=False, support_fraction=None,
                          iqr_range = 1.5, upper_whisker = None):
        '''
        Function to find outliers using the EllipticEnvelope algorithm.
        Makes use of robust statistics to envelop the data inside an elliptic region, assuming it belongs to a 
        gaussian distribution.
        Custom variables:
         - iqr_range: Float number that scales the size of the whiskers, so that it can be adjusted for each cluster or model.
           Defaluts to 1.5 (as commonly used)
         - upper_whisker: Float number to define upper whisker and override the iqr_value, so that the user can specify 
           a hard border instead of depending on the IQR and the distribution of the data.
        https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html?highlight=ellipticenvelope#sklearn.covariance.EllipticEnvelope
        '''
        if(isinstance(self.contamination, float)):
            ee = EllipticEnvelope(store_precision=store_precision, assume_centered=assume_centered, support_fraction=support_fraction, contamination=self.contamination, random_state=self.random_state)
            ee.fit(self.X)
            preds = ee.predict(self.X)
            scores = ee.mahalanobis(self.X)
        elif(isinstance(self.contamination, int)):
            #Let us asume that 5% of the data are outliers
            ee = EllipticEnvelope(store_precision=store_precision, assume_centered=assume_centered, support_fraction=support_fraction, contamination=0.05, random_state=self.random_state)
            ee.fit(self.X)
            scores = ee.mahalanobis(self.X)
            
            #Get top k indices of the scores (higher scores are more anomalous)
            out_idx = scores.argsort()[::-1][:self.contamination]#Argsort sorts from smaller to larger values, we revert it with [::-1]
            preds = np.ones(self.X.shape[0])
            #Assign -1 to outliers
            for idx in out_idx:
                preds[idx] = -1
        else:
            #Let us asume that 5% of the data are outliers
            ee = EllipticEnvelope(store_precision=store_precision, assume_centered=assume_centered, support_fraction=support_fraction, contamination=0.05, random_state=self.random_state)
            ee.fit(self.X)
            scores = ee.mahalanobis(self.X)
            #Find points that are above the boxplot threshold, those are outlier candidates
            Q2 = np.percentile(scores, 50, interpolation = 'midpoint')
            Q1 = np.percentile(scores, 25, interpolation = 'midpoint')
            Q3 = np.percentile(scores, 75, interpolation = 'midpoint')
            IQR = stats.iqr(scores)
            #Upper whisker
            if(upper_whisker == None):
                uw = Q3 + iqr_range*IQR
                print(f'Contamination not specified, finding outliers above Q3 + {iqr_range}*IQR')
            else:
                uw = upper_whisker
                print(f'Contamination not specified, finding outliers above custom upper whisker: {uw}')
            print(f'Q1: {Q1}, Q2: {Q2}, Q3: {Q3}, Upper whisker: ({uw})')
            #Convert into format 1 => Inlier, -1 => Outlier
            preds = np.array([1 if s <= uw else -1 for s in scores])
        
        #Save model as binary file
        with open(self.savepath + '/ellipticenvelope.pkl', 'wb') as f:
            pkl.dump(ee, f)
        
        return(preds, scores)
    
    def elliptic_envelope_predict(self, data, topn: int = None, iqr_range = 1.5, upper_whisker = None):
        '''
        Function used to detect outliers given a pre-trained elliptic envelope model.
        The data must still be either a pandas dataframe or numpy array, with normalized numerical variables.
        If a PCA or SVD was specified when creating the find_outliers class, the data will be transformed accordingly.
        Input:
         - data: Dataframe or numpy array with pre-processed variables (i. e. numerical variables have been scaled,
           standardized or encoded)
         - topn: Integer indicating expected amount of outliers for the test sample. If not specified, will take the one
           provided when declaring the find_outliers class. It will only be used if an integer was assigned to
           the contamination when instantiating the class, otherwise will be ignored.
         Custom variables:
         - iqr_range: Float number that scales the size of the whiskers, so that it can be adjusted for each cluster or model.
           Defaluts to 1.5 (as commonly used)
         - upper_whisker: Float number to define upper whisker and override the iqr_value, so that the user can specify 
           a hard border instead of depending on the IQR and the distribution of the data.
        '''
        #Check data type
        if(isinstance(data, pd.core.frame.DataFrame)):
            X = np.array(data)
        elif(isinstance(data, np.ndarray)):
            X = data
        else:
            raise('data must be either a pandas dataframe or a numpy array')
        #Apply transform if necessary
        if(self.decomposition == 'PCA'):
            X = self.PCA.transform(X)
        elif(self.decomposition == 'SVD'):
            X = self.SVD.transform(X)
        #Load elliptic envelope model
        with open(self.savepath + '/ellipticenvelope.pkl', 'rb') as f:
            ee = pkl.load(f)
        #Handle anomaly calculation
        if(isinstance(self.contamination, float)):
            preds = ee.predict(X)
            scores = ee.mahalanobis(X)
        elif(isinstance(self.contamination, int)):
            if(topn == None):
                topn = self.contamination
            scores = ee.mahalanobis(X)
            
            #Get top k indices of the scores (higher scores are more anomalous)
            out_idx = scores.argsort()[::-1][:topn]#Argsort sorts from smaller to larger values, we revert it with [::-1]
            preds = np.ones(X.shape[0])
            #Assign -1 to outliers
            for idx in out_idx:
                preds[idx] = -1
        else:
            scores_aux = ee.mahalanobis(self.X)
            scores = ee.mahalanobis(X)
            #Find points that are above the boxplot threshold, those are outlier candidates
            Q2 = np.percentile(scores_aux, 50, interpolation = 'midpoint')
            Q1 = np.percentile(scores_aux, 25, interpolation = 'midpoint')
            Q3 = np.percentile(scores_aux, 75, interpolation = 'midpoint')
            IQR = stats.iqr(scores_aux)
            #Upper whisker
            if(upper_whisker == None):
                uw = Q3 + iqr_range*IQR
                print(f'Contamination not specified, finding outliers above Q3 + {iqr_range}*IQR')
            else:
                uw = upper_whisker
                print(f'Contamination not specified, finding outliers above custom upper whisker: {uw}')
            print(f'Q1: {Q1}, Q2: {Q2}, Q3: {Q3}, Upper whisker: ({uw})')
            #Convert into format 1 => Inlier, -1 => Outlier
            preds = np.array([1 if s <= uw else -1 for s in scores])
        return(preds, scores)
    
    def get_elliptic_envelope(self):
        '''
        Method that returns the pre-trained elliptic envelope model.
        '''
        path = self.savepath + '/ellipticenvelope.pkl'
        if(os.path.isfile(path)):
            with open(path, 'rb') as f:
                ee = pkl.load(f)
        else:
            warnings.warn('No pre-trained model found, returning empty object.')
            ee = None
        return(ee)
    
    def isolation_forest(self, n_estimators=100, max_samples='auto', max_features=1.0, bootstrap=False,
                         n_jobs=1, verbose=0, warm_start=False, iqr_range = 1.5, lower_whisker = None):
        '''
        Function to find outliers using the IsolationForest algorithm.
        Makes use of decision trees to randomly split the data and isolate uncommon data points.
        Custom variables:
         - iqr_range: Float number that scales the size of the whiskers, so that it can be adjusted for each cluster or model.
           Defaluts to 1.5 (as commonly used)
         - lower_whisker: Float number to define lower whisker and override the iqr_value, so that the user can specify 
           a hard border instead of depending on the IQR and the distribution of the data.
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest
        '''
        if(isinstance(self.contamination, float)):
            IF = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=self.contamination,
                                 max_features=max_features, bootstrap=bootstrap, n_jobs=n_jobs,
                                 random_state=self.random_state, verbose=verbose,warm_start=warm_start)
            IF.fit(self.X)
            preds = IF.predict(self.X)
            scores = IF.decision_function(self.X)
        elif(isinstance(self.contamination, int)):
            #Let the model estimate the total amount of outliers
            IF = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination='auto',
                                 max_features=max_features, bootstrap=bootstrap, n_jobs=n_jobs,
                                 random_state=self.random_state, verbose=verbose,warm_start=warm_start)
            IF.fit(self.X)
            scores = IF.decision_function(self.X)
            
            #Get top k indices of the scores (lower scores are more anomalous)
            out_idx = scores.argsort()[:self.contamination]#Argsort sorts from smaller to larger values
            preds = np.ones(self.X.shape[0])
            #Assign -1 to outliers
            for idx in out_idx:
                preds[idx] = -1
        else:
            #Let the model estimate the total amount of outliers
            IF = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination='auto',
                                 max_features=max_features, bootstrap=bootstrap, n_jobs=n_jobs,
                                 random_state=self.random_state, verbose=verbose,warm_start=warm_start)
            IF.fit(self.X)
            scores = IF.decision_function(self.X)
            #Find points that are above the boxplot threshold, those are outlier candidates
            Q2 = np.percentile(scores, 50, interpolation = 'midpoint')
            Q1 = np.percentile(scores, 25, interpolation = 'midpoint')
            Q3 = np.percentile(scores, 75, interpolation = 'midpoint')
            IQR = stats.iqr(scores)
            #Lower whisker
            if(lower_whisker == None):
                lw = Q1 - iqr_range*IQR
                print(f'Contamination not specified, finding outliers below Q1 - {iqr_range}*IQR')
            else:
                lw = lower_whisker
                print(f'Contamination not specified, finding outliers below custom lower whisker: {lw}')
            print(f'Q1: {Q1}, Q2: {Q2}, Q3: {Q3}, Lower whisker: ({lw})')
            #Convert into format 1 => Inlier, -1 => Outlier
            preds = np.array([1 if s >= lw else -1 for s in scores])
        
        #Save model as binary file
        with open(self.savepath + '/isolationforest.pkl', 'wb') as f:
            pkl.dump(IF, f)
        return(preds, scores)

    def isolation_forest_predict(self, data, topn: int = None, iqr_range = 1.5, lower_whisker = None):
        '''
        Function used to detect outliers given a pre-trained isolation forest model.
        The data must still be either a pandas dataframe or numpy array, with normalized numerical variables.
        If a PCA or SVD was specified when creating the find_outliers class, the data will be transformed accordingly.
        Input:
         - data: Dataframe or numpy array with pre-processed variables (i. e. numerical variables have been scaled,
           standardized or encoded)
         - topn: Integer indicating expected amount of outliers for the test sample. If not specified,
           will take the one provided when declaring the find_outliers class. It will only be used if an integer was
           assigned to the contamination when instantiating the class, otherwise will be ignored.
         Custom variables:
         - iqr_range: Float number that scales the size of the whiskers, so that it can be adjusted for each cluster or model.
           Defaluts to 1.5 (as commonly used)
         - lower_whisker: Float number to define lower whisker and override the iqr_value, so that the user can specify 
           a hard border instead of depending on the IQR and the distribution of the data.
        '''
        #Check data type
        if(isinstance(data, pd.core.frame.DataFrame)):
            X = np.array(data)
        elif(isinstance(data, np.ndarray)):
            X = data
        else:
            raise('data must be either a pandas dataframe or a numpy array')
        #Apply transform if necessary
        if(self.decomposition == 'PCA'):
            X = self.PCA.transform(X)
        elif(self.decomposition == 'SVD'):
            X = self.SVD.transform(X)
        #Load elliptic envelope model
        with open(self.savepath + '/isolationforest.pkl', 'rb') as f:
            IF = pkl.load(f)
        #Handle anomaly calculation
        if(isinstance(self.contamination, float)):
            preds = IF.predict(X)
            scores = IF.decision_function(X)
        elif(isinstance(self.contamination, int)):
            if(topn == None):
                topn = self.contamination
            scores = IF.decision_function(X)
            
            #Get top k indices of the scores (lower scores are more anomalous)
            out_idx = scores.argsort()[:topn]#Argsort sorts from smaller to larger values
            preds = np.ones(X.shape[0])
            #Assign -1 to outliers
            for idx in out_idx:
                preds[idx] = -1
        else:
            scores_aux = IF.decision_function(self.X)
            scores = IF.decision_function(X)
            #Find points that are above the boxplot threshold, those are outlier candidates
            Q2 = np.percentile(scores_aux, 50, interpolation = 'midpoint')
            Q1 = np.percentile(scores_aux, 25, interpolation = 'midpoint')
            Q3 = np.percentile(scores_aux, 75, interpolation = 'midpoint')
            IQR = stats.iqr(scores_aux)
            #Lower whisker
            if(lower_whisker == None):
                lw = Q1 - iqr_range*IQR
                print(f'Contamination not specified, finding outliers below Q1 - {iqr_range}*IQR')
            else:
                lw = lower_whisker
                print(f'Contamination not specified, finding outliers below custom lower whisker: {lw}')
            print(f'Q1: {Q1}, Q2: {Q2}, Q3: {Q3}, Lower whisker: ({lw})')
            #Convert into format 1 => Inlier, -1 => Outlier
            preds = np.array([1 if s >= lw else -1 for s in scores])
        return(preds, scores)
    
    def get_isolation_forest(self):
            '''
            Method that returns the trained isolation forest model.
            '''
            path = self.savepath + '/isolationforest.pkl'
            if(os.path.isfile(path)):
                with open(path, 'rb') as f:
                    IF = pkl.load(f)
            else:
                warnings.warn('No pre-trained model found, returning empty object.')
                IF = None
            return(IF)
    
    def oc_svm(self, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5,
               shrinking=True, cache_size=200, verbose=False, max_iter=- 1, iqr_range = 1.5, lower_whisker = None):
        '''
        Function to find outliers using the One-Class SVM algorithm, even though it is more suitable to perform novelty detection.
        This algorithm attempts to model the high-dimensional distribution of the data. 
        Works best when the original dataset has no (or very few) outliers. Also does not use a random state.
        Custom variables:
         - iqr_range: Float number that scales the size of the whiskers, so that it can be adjusted for each cluster or model.
           Defaluts to 1.5 (as commonly used)
         - lower_whisker: Float number to define lower whisker and override the iqr_value, so that the user can specify 
           a hard border instead of depending on the IQR and the distribution of the data.
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest
        '''
        if(isinstance(self.contamination, float)):
            ocsvm = OneClassSVM(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, nu=nu,
               shrinking=shrinking, cache_size=cache_size, verbose=verbose, max_iter=- max_iter)
            ocsvm.fit(self.X)
            preds = ocsvm.predict(self.X)
            scores = ocsvm.decision_function(self.X)
        elif(isinstance(self.contamination, int)):
            #Let the model estimate the total amount of outliers
            ocsvm = OneClassSVM(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, nu=nu,
               shrinking=shrinking, cache_size=cache_size, verbose=verbose, max_iter=- max_iter)
            ocsvm.fit(self.X)
            scores = ocsvm.decision_function(self.X)
            
            #Get top k indices of the scores (lower scores are more anomalous)
            out_idx = scores.argsort()[:self.contamination]#Argsort sorts from smaller to larger values
            preds = np.ones(self.X.shape[0])
            #Assign -1 to outliers
            for idx in out_idx:
                preds[idx] = -1
        else:
            #Let the model estimate the total amount of outliers
            ocsvm = OneClassSVM(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, nu=nu,
               shrinking=shrinking, cache_size=cache_size, verbose=verbose, max_iter=- max_iter)
            ocsvm.fit(self.X)
            scores = ocsvm.decision_function(self.X)
            #Find points that are above the boxplot threshold, those are outlier candidates
            Q2 = np.percentile(scores, 50, interpolation = 'midpoint')
            Q1 = np.percentile(scores, 25, interpolation = 'midpoint')
            Q3 = np.percentile(scores, 75, interpolation = 'midpoint')
            IQR = stats.iqr(scores)
            #Lower whisker
            if(lower_whisker == None):
                lw = Q1 - iqr_range*IQR
                print(f'Contamination not specified, finding outliers below Q1 - {iqr_range}*IQR')
            else:
                lw = lower_whisker
                print(f'Contamination not specified, finding outliers below custom lower whisker: {lw}')
            print(f'Q1: {Q1}, Q2: {Q2}, Q3: {Q3}, Lower whisker: ({lw})')
            #Convert into format 1 => Inlier, -1 => Outlier
            preds = np.array([1 if s >= lw else -1 for s in scores])
        
        #Save model as binary file
        with open(self.savepath + '/ocsvm.pkl', 'wb') as f:
            pkl.dump(ocsvm, f)
        return(preds, scores)

    def oc_svm_predict(self, data, topn: int = None, iqr_range = 1.5, lower_whisker = None):
        '''
        Function used to detect outliers given a pre-trained isolation forest model.
        The data must still be either a pandas dataframe or numpy array, with normalized numerical variables.
        If a PCA or SVD was specified when creating the find_outliers class, the data will be transformed accordingly.
        Input:
         - data: Dataframe or numpy array with pre-processed variables (i. e. numerical variables have been scaled,
           standardized or encoded)
         - topn: Integer indicating expected amount of outliers for the test sample. If not specified,
           will take the one provided when declaring the find_outliers class. It will only be used if an integer
           was assigned to the contamination when instantiating the class, otherwise will be ignored.
         Custom variables:
         - iqr_range: Float number that scales the size of the whiskers, so that it can be adjusted for each cluster or model.
           Defaluts to 1.5 (as commonly used)
         - lower_whisker: Float number to define lower whisker and override the iqr_value, so that the user can specify 
           a hard border instead of depending on the IQR and the distribution of the data.
        '''
        #Check data type
        if(isinstance(data, pd.core.frame.DataFrame)):
            X = np.array(data)
        elif(isinstance(data, np.ndarray)):
            X = data
        else:
            raise('data must be either a pandas dataframe or a numpy array')
        #Apply transform if necessary
        if(self.decomposition == 'PCA'):
            X = self.PCA.transform(X)
        elif(self.decomposition == 'SVD'):
            X = self.SVD.transform(X)
        #Load elliptic envelope model
        with open(self.savepath + '/ocsvm.pkl', 'rb') as f:
            ocsvm = pkl.load(f)
        #Handle anomaly calculation
        if(isinstance(self.contamination, float)):
            preds = ocsvm.predict(X)
            scores = ocsvm.decision_function(X)
        elif(isinstance(self.contamination, int)):
            if(topn == None):
                topn = self.contamination
            scores = ocsvm.decision_function(X)
            
            #Get top k indices of the scores (lower scores are more anomalous)
            out_idx = scores.argsort()[:topn]#Argsort sorts from smaller to larger values
            preds = np.ones(X.shape[0])
            #Assign -1 to outliers
            for idx in out_idx:
                preds[idx] = -1
        else:
            scores_aux = ocsvm.decision_function(self.X)
            scores = ocsvm.decision_function(X)
            #Find points that are above the boxplot threshold, those are outlier candidates
            Q2 = np.percentile(scores_aux, 50, interpolation = 'midpoint')
            Q1 = np.percentile(scores_aux, 25, interpolation = 'midpoint')
            Q3 = np.percentile(scores_aux, 75, interpolation = 'midpoint')
            IQR = stats.iqr(scores_aux)
            #Lower whisker
            if(lower_whisker == None):
                lw = Q1 - iqr_range*IQR
                print(f'Contamination not specified, finding outliers below Q1 - {iqr_range}*IQR')
            else:
                lw = lower_whisker
                print(f'Contamination not specified, finding outliers below custom lower whisker: {lw}')
            print(f'Q1: {Q1}, Q2: {Q2}, Q3: {Q3}, Lower whisker: ({lw})')
            #Convert into format 1 => Inlier, -1 => Outlier
            preds = np.array([1 if s >= lw else -1 for s in scores])
        return(preds, scores)
    
    def get_oc_svm(self):
            '''
            Method that returns the trained one-class SVM model.
            '''
            path = self.savepath + '/ocsvm.pkl'
            if(os.path.isfile(path)):
                with open(path, 'rb') as f:
                    ocsvm = pkl.load(f)
            else:
                warnings.warn('No pre-trained model found, returning empty object.')
                ocsvm = None
            return(ocsvm)
    #@staticmethod
    #def mydist(x, y):
    #    return np.sum((x-y)**2)
    
    def local_outlier_factor(self, n_neighbors=20, algorithm='auto', leaf_size=30, metric = None,
                             p=2, metric_params=None, contamination='auto', novelty=True, n_jobs=None,
                             iqr_range = 1.5, lower_whisker = None):
        '''
        Function to find outliers using the Local Outlier Factor algorithm.
        Works remarkably well for regions with different densities in the data, as it makes use of the nearest
        neighbors to compute the scores.
        Changing novelty to false will result in an error.
        Custom variables:
         - iqr_range: Float number that scales the size of the whiskers, so that it can be adjusted for each cluster or model.
           Defaluts to 1.5 (as commonly used)
         - lower_whisker: Float number to define lower whisker and override the iqr_value, so that the user can specify 
           a hard border instead of depending on the IQR and the distribution of the data.
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
        '''
        #Use pre-computed metric if none was specified
        if(metric == None):
            metric = 'euclidean'
            warnings.warn('No metric specified, defaulting to euclidean distance')
        if(isinstance(self.contamination, float)):
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, metric=metric,
                             p=p, metric_params=metric_params, contamination=self.contamination, novelty=novelty, n_jobs=n_jobs)
            lof.fit(self.X)
            preds = lof.predict(self.X)
            scores = lof.decision_function(self.X)
        elif(isinstance(self.contamination, int)):
            #Let the model estimate the total amount of outliers
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, metric=metric,
                             p=p, metric_params=metric_params, contamination='auto', novelty=novelty, n_jobs=n_jobs)
            lof.fit(self.X)
            scores = lof.decision_function(self.X)
            
            #Get top k indices of the scores (lower scores are more anomalous)
            out_idx = scores.argsort()[:self.contamination]#Argsort sorts from smaller to larger values
            preds = np.ones(self.X.shape[0])
            #Assign -1 to outliers
            for idx in out_idx:
                preds[idx] = -1
        else:
            #Let the model estimate the total amount of outliers
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, metric=metric,
                             p=p, metric_params=metric_params, contamination='auto', novelty=novelty, n_jobs=n_jobs)
            lof.fit(self.X)
            #scores = lof.negative_outlier_factor_
            scores = lof.decision_function(self.X)
            #Find points that are above the boxplot threshold, those are outlier candidates
            Q2 = np.percentile(scores, 50, interpolation = 'midpoint')
            Q1 = np.percentile(scores, 25, interpolation = 'midpoint')
            Q3 = np.percentile(scores, 75, interpolation = 'midpoint')
            IQR = stats.iqr(scores)
            #Lower whisker
            if(lower_whisker == None):
                lw = Q1 - iqr_range*IQR
                print(f'Contamination not specified, finding outliers below Q1 - {iqr_range}*IQR')
            else:
                lw = lower_whisker
                print(f'Contamination not specified, finding outliers below custom lower whisker: {lw}')
            print(f'Q1: {Q1}, Q2: {Q2}, Q3: {Q3}, Lower whisker: ({lw})')
            #Convert into format 1 => Inlier, -1 => Outlier
            preds = np.array([1 if s >= lw else -1 for s in scores])
        
        #Save model as binary file
        with open(self.savepath + '/lof.pkl', 'wb') as f:
            pkl.dump(lof, f)
        return(preds, scores)
    
    def local_outlier_factor_predict(self, data, topn: int = None, iqr_range = 1.5, lower_whisker = None):
        '''
        Function used to detect outliers given a pre-trained isolation forest model.
        The data must still be either a pandas dataframe or numpy array, with normalized numerical variables.
        If a PCA or SVD was specified when creating the find_outliers class, the data will be transformed accordingly.
        Input:
         - data: Dataframe or numpy array with pre-processed variables (i. e. numerical variables have been scaled,
           standardized or encoded)
         - topn: Integer indicating expected amount of outliers for the test sample. If not specified,
           will take the one provided when declaring the find_outliers class. It will only be used if an
           integer was assigned to the contamination when instantiating the class, otherwise will be ignored.
         Custom variables:
         - iqr_range: Float number that scales the size of the whiskers, so that it can be adjusted for each cluster or model.
           Defaluts to 1.5 (as commonly used)
         - lower_whisker: Float number to define lower whisker and override the iqr_value, so that the user can specify 
           a hard border instead of depending on the IQR and the distribution of the data.
        '''
        #Check data type
        if(isinstance(data, pd.core.frame.DataFrame)):
            X = np.array(data)
        elif(isinstance(data, np.ndarray)):
            X = data
        else:
            raise('data must be either a pandas dataframe or a numpy array')
        #Apply transform if necessary
        if(self.decomposition == 'PCA'):
            X = self.PCA.transform(X)
        elif(self.decomposition == 'SVD'):
            X = self.SVD.transform(X)
        #Load elliptic envelope model
        with open(self.savepath + '/lof.pkl', 'rb') as f:
            lof = pkl.load(f)
        #Handle anomaly calculation
        if(isinstance(self.contamination, float)):
            preds = lof.predict(X)
            scores = lof.decision_function(X)
        elif(isinstance(self.contamination, int)):
            if(topn == None):
                topn = self.contamination
            scores = lof.decision_function(X)
            
            #Get top k indices of the scores (lower scores are more anomalous)
            out_idx = scores.argsort()[:topn]#Argsort sorts from smaller to larger values
            preds = np.ones(X.shape[0])
            #Assign -1 to outliers
            for idx in out_idx:
                preds[idx] = -1
        else:
            scores_aux = lof.decision_function(self.X)
            scores = lof.decision_function(X)
            #Find points that are above the boxplot threshold, those are outlier candidates
            Q2 = np.percentile(scores_aux, 50, interpolation = 'midpoint')
            Q1 = np.percentile(scores_aux, 25, interpolation = 'midpoint')
            Q3 = np.percentile(scores_aux, 75, interpolation = 'midpoint')
            IQR = stats.iqr(scores_aux)
            #Lower whisker
            if(lower_whisker == None):
                lw = Q1 - iqr_range*IQR
                print(f'Contamination not specified, finding outliers below Q1 - {iqr_range}*IQR')
            else:
                lw = lower_whisker
                print(f'Contamination not specified, finding outliers below custom lower whisker: {lw}')
            print(f'Q1: {Q1}, Q2: {Q2}, Q3: {Q3}, Lower whisker: ({lw})')
            #Convert into format 1 => Inlier, -1 => Outlier
            preds = np.array([1 if s >= lw else -1 for s in scores])
        return(preds, scores)
    
    def get_local_outlier_factor(self):
            '''
            Method that returns the trained local outlier factor model.
            '''
            path = self.savepath + '/lof.pkl'
            if(os.path.isfile(path)):
                with open(path, 'rb') as f:
                    lof = pkl.load(f)
            else:
                warnings.warn('No pre-trained model found, returning empty object.')
                lof = None
            return(lof)