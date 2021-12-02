# Libraries import

from dataclasses import dataclass
from typing import Union

# Basic Numerical Libraries
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.spatial import distance
from scipy.stats import gamma as gamma_dist
from tensorflow_probability.python.math.psd_kernels.internal.util \
    import pairwise_square_distance_matrix

# Basic Preprocessing, Dataset Libraries
from sklearn import __version__ as sklearn_version
from sklearn.datasets import make_blobs, load_iris, load_breast_cancer, load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#Custom part
from sklearn.neighbors import KNeighborsClassifier

# Basic Plotting Libraries
import matplotlib
import matplotlib.pyplot as plt

# Basic Data Wrangling Libraries
from pandas import Series

# Basic Scoring Libraries
import sklearn.metrics as skm


print(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))} .\n")
print(f"The tensorflow version is {tf.__version__}.\n")
#assert sklearn_version == '0.23.0', "We need sklearn 0.23.0 to get the cluster centers from toy-datasets."
print(f"The scikit-learn version is {sklearn_version}.\n")
print(f"The matplotlib version is {matplotlib.__version__}.\n")

@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None,None], dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.int32)
        ]
    )
def bot_k(distance_matrix, K):

    tf.assert_greater(K, 0, message='K must be at least 1')
    topv, topi = tf.math.top_k(-distance_matrix, k=K+1)
    topv = -topv[:,1:]
    topi = topi[:,1:]
    return topv, topi


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None,None], dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.int32)
        ]
    )
def std_knn(distance_matrix, K):
    topv, _ = bot_k(distance_matrix, K)
    return tf.math.reduce_std(topv, axis=1)


# Auxiliary Functions
# _matrix: returns a matrix
# _tensor: returns a tensor-not-matrix


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None,None], dtype=tf.float64)
        ]
    )
def square_distance_matrix(data_matrix):
    return pairwise_square_distance_matrix(data_matrix, data_matrix, feature_ndims=1)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None,None], dtype=tf.float64)
        ]
    )
def pairwise_difference_tensor(data_matrix):
    a = tf.expand_dims(data_matrix, axis=1)
    b = tf.expand_dims(data_matrix, axis=0)
    diff = tf.add(a,-b)
    return diff

@tf.function#(input_signature=(tf.TensorSpec(shape=[None,None], dtype=tf.float64),tf.TensorSpec(shape=None, dtype=tf.bool)))
def sum_over_cols(matrix, keepdims=True): # OK # OJO! no son inversas
    # sum_over_cols keepdims = True, reproduce matrix_to_vector
    return tf.reduce_sum(matrix, axis=1, keepdims=keepdims)

@tf.function
def exp_matrix(m, gamma):
    return tf.exp(-tf.math.multiply(gamma,m))


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None,None], dtype=tf.float64)
        ]
    )
@tf.function
def increase_rank(matrix): # OK
    return tf.expand_dims(matrix, axis=2)

@tf.function
def expected_value_matrix(function_matrix, wave_function_matrix): # OK
    if tf.rank(function_matrix) == 3:
        wave_function_matrix = increase_rank(wave_function_matrix)
    vec_top = tf.multiply(wave_function_matrix, function_matrix)
    vec_bot = sum_over_cols(wave_function_matrix)
    return tf.math.divide(vec_top, vec_bot)

# calculo de potencial

## calculo de wavefunction
@tf.function
def wave_function_matrix(m, gamma):
    return exp_matrix(square_distance_matrix(m), gamma)

@tf.function
def wave_function(m, gamma): # OK
    return sum_over_cols(wave_function_matrix(m,gamma))

@tf.function
def potential_matrix(m, gamma): # OK
    foo_matrix = tf.math.multiply(gamma,square_distance_matrix(m))
    wf_matrix = wave_function_matrix(m, gamma)
    expc_foo_matrix = expected_value_matrix(foo_matrix, wf_matrix)
    return expc_foo_matrix

@tf.function
def potential(m, gamma): # OK
    return sum_over_cols(potential_matrix(m, gamma))

# calculo de gradiente de potencial

@tf.function
def grad_potential(x, gamma): # OK
    wf_matrix = wave_function_matrix(x, gamma)
    pw_diff_tensor = pairwise_difference_tensor(x) #OK
    sq_dist_tensor = increase_rank(square_distance_matrix(x))

    # EXPECTED VALUES
    foo_a = tf.math.multiply(2*tf.expand_dims(gamma, axis=1),pw_diff_tensor)
    expc_a_tensor = expected_value_matrix(foo_a, wf_matrix)
    expc_a_matrix = sum_over_cols(expc_a_tensor, keepdims=False)
    #b potential
    foo_c = tf.math.multiply(2*tf.expand_dims(gamma**2, axis=1),sq_dist_tensor*pw_diff_tensor)
    expc_c_tensor = expected_value_matrix(foo_c, wf_matrix)
    expc_c_matrix = sum_over_cols(expc_c_tensor, keepdims=False)

    pot = potential(x, gamma)

    grad = expc_a_matrix + tf.multiply(pot,expc_a_matrix) - expc_c_matrix

    return pot, grad

def final_allocation(data, step_size):
    clusters = -np.ones((data.shape[0])) # non-allocated datapoints have -1 value
    nonalloc_idx = np.argwhere(clusters == -1) # indexes of non-allocated datapoints
    # alloc of datapoints is given by a distance criteria such that if one datapoint
    # is close enough to another one they will be group together.
    #distances = distance.squareform(distance.pdist(data))
    distances = tf.math.sqrt(square_distance_matrix(data)).numpy()
    c = 0 # counter and the value of cluster classes
    # while there are non-allocated datapoints the loop will continue
    while nonalloc_idx.shape[0]>0:
        # do
        i = nonalloc_idx[0] # counter for searching rows
        # allocation
        clusters[nonalloc_idx[distances[i, nonalloc_idx] <= 3*step_size]] = c
        # counter update
        c += 1
        nonalloc_idx = np.argwhere(clusters == -1)

    return clusters

def stop_condition(info, tol):
    next_x = info[-1]['coord']
    previous_x = info[-2]['coord']
    diff = next_x - previous_x
    return tf.experimental.numpy.allclose(diff, tf.zeros_like(diff), atol=tol)



def grad_descent(start, gamma, lr, tol, max_iter=500):

    info = []
    x = start
    if not tf.is_tensor(x):
        x = tf.convert_to_tensor(x, dtype=tf.float64)
    if not tf.is_tensor(gamma):
        gamma = tf.convert_to_tensor(gamma, dtype=tf.float64)
    step = 50#int(max_iter/10) #Custom part
    for i in range(1, max_iter): # from 1 to max_iter avoids IF i not 0
        if i%step == 0:
            print('step: ', i)
            if stop_condition(info, tol):
                print("No more iterations needed.")
                break
        v, dv = grad_potential(x, gamma)
        info.append({
            'iter': i,
            'coord': x,
            'pot': v,
            'gpot': dv
            })
        x = x - lr*dv
    #if verbose:

    
    return info




def class_reassignment(classes, dictionary):
    classes = Series(classes, dtype=int)
    return classes.replace(dictionary)


def sigma_calc(kind, X):
    if isinstance(kind, str):
        try:
            v = float(kind.split(".")[1])/(10**len(kind.split(".")[1]))

            k_int = tf.cast(tf.constant(X.shape[0]*v), tf.int32)

            Dx = square_distance_matrix(X)
        except:
            print("sigma con only be a float, knn.v or gamma.v where v are integers.")

        if kind.split('.')[0] == "knn":
            # knn.v
            # density-based calculation of sigma (variable)
            n = X.shape[0]
            knn, _ = bot_k(Dx, k_int)
            # si sum_k=1_knn (dist(x-xk))/n

            s = sum_over_cols(tf.math.sqrt(knn))/tf.cast(k_int, tf.float64)

            return s
        else:
            # gamma.v
            # fixed value calculation with gamma distribution
            
            std_k = std_knn(Dx, K=k_int)
            fit_alpha, _, fit_beta = gamma_dist.fit(std_k.numpy())
            sigma = np.array(np.repeat([fit_alpha*fit_beta],X.shape[0]))
    else:
        # fixed constant value
        sigma = np.array(np.repeat([kind],X.shape[0]))

    return sigma




@dataclass
class Qluster():
    sigma: Union[float, str] = 'gamma.4'
    max_iter: int = 1000
    learning_rate = "constant" # rn only one option
    learning_rate_init = 0.01
    labels_ = None
    data = None
    sigma_value = None
    #Custom part
    classifier = None

        
    def fit(self, X, tol=0.001, step_size=0.5,
            n_neighbors = 3, metric = 'minkowski', weights = 'uniform', algorithm = 'auto', leaf_size = 30, p = 2):
        self.sigma_value = sigma_calc(self.sigma, X)
        gamma = 1/(2*self.sigma_value**2)
        gamma =  tf.constant(gamma, dtype=tf.float64)
        #gamma = tf.expand_dims(gamma, axis=1)
        print(gamma.shape)
        self.data = grad_descent(X, gamma, self.learning_rate_init, tol, max_iter=self.max_iter)
        self.labels_ = final_allocation(self.data[-1]['coord'], step_size=step_size)
        #Custom part
        self.classifier = KNeighborsClassifier(n_neighbors = n_neighbors, metric = metric, weights = weights,
                                               algorithm = algorithm, leaf_size = leaf_size, p = p).fit(X, self.labels_)
    

    def score(self, metric, y=None):
        if metric == 'jaccard':
            scr = self.__score_jaccard(y)
        elif metric == 'vmeasure':
            scr = self.__score_vmeasure(y)
        elif metric == 'sil':
            scr = self.__score_silhouette()
        else:
            pass
        return scr
    
    def predict(self, X, return_probs = False):
        if(return_probs == False):
            labels = self.classifier.predict(X)
            return labels
        else:
            labels = self.classifier.predict(X)
            labels_probs = self.classifier.predict_proba(X)
            return labels, labels_probs
    
    # wrappers score
    def __score_jaccard(self, y, average='micro'):
        """The Jaccard index [1], or Jaccard similarity coefficient, defined as the size 
        of the intersection divided by the size of the union of two label sets, is used 
        to compare set of predicted labels for a sample to the corresponding set of labels
        in y_true."""
        return skm.jaccard_score(y, self.labels_, average=average)
    
    def __score_vmeasure(self, y):
        return skm.homogeneity_completeness_v_measure(y, self.labels_)
    
    def __score_silhouette(self, metric='euclidean'):
        skm.silhouette_score(self.data[-1]['coord'], self.labels_, metric=metric)
    # etc
