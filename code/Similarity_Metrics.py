
# coding: utf-8

# # CS579: Online Social Network Analysis
# 
# # Final Project - Recommendation Systems Using Map-Reduce
# 
# 
# $$J V P S Avinash $$ <br>
# $$Rakshith Muniraju $$
# 
# 

# # Part - 3 : Computing Similarity Measures

# This file hosts the calculation of various similarity measures for movie pair. <br>
# We have covered Cosine Similarity, Jaccard Similarity, Generalized Jaccard, Pearson's Correlation and its Normalization. <br>
# This file is downloaded as similarity_metrics.py and this file is loaded in Part-2 to calculate Similarity between two movies.

# In[1]:

import math
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.metrics import jaccard_similarity_score

import itertools 


# In[2]:

def norm(vector):
    """
    Computes the Euclidean norm of one row of given vector.
    """
    return np.linalg.norm(vector)


# # # Cosine Similarity
# # ![cosine.png](cosine.png)

# In[8]:

def cosine(v1, v2):
    """
    Computes the cosine similarity between two vectors.
    Generally, we divide the dot product of two vectors by the product of norm of those two vectors.
    """
    numerator = np.dot(csr_matrix(v1).toarray(),csr_matrix(v2).T.toarray())
    denominator = (norm(v1) * norm(v2))
    return (numerator / denominator) if denominator else 0.0


# # # Jaccard Similarity
# 
# # ![jaccard.png](jaccard.png)

# In[9]:

def jaccard(v1,v2):
    """
    Computes the Jaccard similarity between two vectors.
    Generally, we divide the number of intersection values between two vectors by the number of union values between two vectors
    """
    countlistA = Counter(v1)
    countlistB = Counter(v2)
    intersection_of_two_vectors = len(list((countlistB & countlistA).elements()))
    union_of_two_vectors = len(v1) + len(v2)
    return (intersection_of_two_vectors / float(union_of_two_vectors)) if union_of_two_vectors else 0.0


# # # Generalized Jaccard Similarity
# 
# # ![gen_jaccard.png](gen_jaccard.png)

# In[5]:

def generalized_jaccard(v1,v2):
    """
    Computes the Generalized Jaccard similarity between two vectors.
    Generally, we divide the minimum of values between two vectors by maximum values between two vectors.
    """
    sum_min = sum_max = 0.0
    for x,y in zip(v1,v2):
        sum_min +=  min(x,y)
        sum_max +=  max(x,y)
    return sum_min / sum_max if sum_max else 0.0


# # # Pearson's Correlation 
# ![pearson-300x156.gif.png](pearson-300x156.gif.png)
#  where n is number of items in a vector 

# In[10]:

def pearson_correlation(v1,v2):
    """
    Computes the Pearson's Correlation between two vectors.
    The correlation coefficient ranges from −1 to 1. 
    A value of 1 implies that a linear equation describes the relationship between X and Y perfectly,
    with all data points lying on a line for which Y increases as X increases. 
    A value of −1 implies that all data points lie on a line for which Y decreases as X increases. 
    A value of 0 implies that there is no linear correlation between the variables.
    """
    n = len(v1)
    sigma_xy = np.dot(csr_matrix(v1).toarray(),csr_matrix(v2).T.toarray())
    sigma_x = sum(v1)
    sigma_y = sum(v2)
    sigma_x2 = norm(v1) * norm(v1)
    sigma_y2 = norm(v2) * norm(v2)
    numerator = (n*sigma_xy - (sigma_x * sigma_y))
    denominator1 = (n*sigma_x2) - (sigma_x * sigma_x)
    denominator2 = (n*sigma_y2) - (sigma_y * sigma_y)
    if denominator1 > 0.0:
        denominator1 = math.sqrt(denominator1)
    else:
        denominator1 = 0.0
    if denominator2 > 0.0:
        denominator2 = math.sqrt(denominator2)
    else:
        denominator2 = 0.0
    denominator = denominator1 * denominator2
    return (numerator / denominator) if denominator else 0.0


# In[11]:

def normalized_pearson_correlation(v1,v2):
    """
    We normalize the value from the above pearson's coefficient in between 0 to 0.5 and the maximum value 
    being 0.5
    """
    return (pearson_correlation(v1,v2) + 1.0) / 2.0 

