
# # Car consumption prediction project 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib as mpl
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from matplotlib.pyplot import figure
from sklearn import cluster
from sklearn.cluster import KMeans 
from sklearn.metrics import completeness_score, homogeneity_score
from sklearn.naive_bayes import GaussianNB
import pickle


# #### here we read the data using the pandas library




df = pd.read_csv('mpgTrainingSet-headings.csv')


#getting the features that have numerical values
continuous_feature = [features for features in df.columns if df[features].dtypes != 'O']


#this function separates the features from the target, we excleded cubic_inch because it increased out accuracy
def loading(df):
    feature_scale=[feature for feature in df.columns if feature not in ['Consumption','brand', 'car_name','cubic_inch']]
    return df[feature_scale] , df['Consumption']


#
#we load the test data 
df_test = pd.read_csv('mpgTestSet-headings.csv')



#we use the loading function written earlier to separate the features from the target 
X_train, y_train = loading(df) 
X_test, y_test = loading(df_test)


# #### we use the minmax scaller function to normalize the train data using fit_transform() function <br> then we use same normalization parameters to transform the test data using the transform() function

scaler=MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
sc = scaler


#we take a brief look at the training features after scaling them down
feature_scale=[feature for feature in df.columns if feature not in ['Consumption','brand', 'car_name','cubic_inch']]
pd.DataFrame(X_train, columns=feature_scale).head(5)



#LDR

LDR = LinearDiscriminantAnalysis()
LDR.fit(X_train, y_train)
ldy_pred = LDR.predict(X_test)




filename = "models/model-lin.b"


# In[49]:


pickle.dump((sc, LDR), open(filename,'wb'))


