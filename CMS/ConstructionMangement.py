# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:11:28 2021

@author: Danylo
"""

%reset -f

#%% Code for Construction Management Software
# each section should be own function/code to call

#%% Read in Data
import numpy as np
import pandas as pd
import pickle
import os


infile = "Construction - DPR Plan- Aug.xlsx"
df = pd.read_excel(infile)




#%% Exploratory Data Analysis




#%% Data Wrangling
# remove unwanted columns
df.columns
df = df.drop(columns=['S.NO.', 'PROGRAM GROUP', 'ACTIVITY ID','YEAR', 'MONTH'])

# combine all text to one column
df['TRANSFORMATION AND DESCRIPTION'] = df['TRANSFORMATION'] + ' ' + df['ACTIVITY DESCRIPTION']

# remove null (zero) values from target 
df = df.replace(0,np.nan)
df = df.dropna(axis=0)
df = df.reset_index()

# create new columns: completion rate days per unit and inverse
from datetime import timedelta,datetime
_startdate = df['START DATE']
_enddate = df['END DATE']
_totaltime = pd.to_datetime(_enddate,format='%d-%m-%Y') - pd.to_datetime(_startdate,format='%d-%m-%Y')
completion_rate_days_per_unit = np.divide(_totaltime,df['PLANNED QTY'].values)
_one_day = pd.Timedelta("1 days")
completion_rate_days_per_unit = np.divide(completion_rate_days_per_unit,_one_day)
completion_rate_units_per_day = np.divide(_one_day,completion_rate_days_per_unit)
df['COMPLETION RATE DAYS PER UNIT'] = completion_rate_days_per_unit
df['COMPLETION RATE UNITS PER DAY'] = completion_rate_units_per_day


# remove unwanted columns again
df = df.drop(columns=['START DATE', 'END DATE', 'PLANNED QTY'])
df = df.drop(columns=['TRANSFORMATION', 'ACTIVITY DESCRIPTION'])

# identify feature and target
feature_names = ['TRANSFORMATION AND DESCRIPTION']
target_names = ['COMPLETION RATE DAYS PER UNIT']


#%% Preprocess
# combine columns transformation and activity description
# remove punctuation
# split numbers from text
# convert roman numerals to number
# remove capitalization
# remove word endings

"""
# tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

strip_accents = 'ascii'
lowercase = True

vectorizer = TfidfVectorizer(strip_accents=strip_accents,lowercase=lowercase)
X = vectorizer.fit_transform(df['TRANSFORMATION AND DESCRIPTION'])

# save tfidf vectorizer
vectorizer_filename= "vectorizer_tfidf.pkl"
with open(vectorizer_filename, 'wb') as outfile:
    pickle.dump(vectorizer,outfile)
"""

# Count Vectorize - easier to search later
from sklearn.feature_extraction.text import CountVectorizer

strip_accents = 'ascii'
lowercase = True
binary = True

vectorizer = CountVectorizer(strip_accents=strip_accents,lowercase=lowercase,binary=binary)
X = vectorizer.fit_transform(df['TRANSFORMATION AND DESCRIPTION'])

# save Vectorizer
vectorizer_filename= "vectorizer_count.pkl"
with open(vectorizer_filename, 'wb') as outfile:
    pickle.dump(vectorizer,outfile)
    
# save X
X_filename= "vectorizer_count_X.pkl"
with open(X_filename, 'wb') as outfile:
    pickle.dump(X,outfile)


#%% Split data
from sklearn.model_selection import train_test_split
X = X
y = df[target_names]

random_state = 42
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)





#%% Machine Learning Model for 'COMPLETION RATE DAYS PER UNIT'

# Random Forest Regressor, MAE=123
from sklearn.ensemble import RandomForestRegressor
n_estimators = 500
random_state = 42
model = RandomForestRegressor(n_estimators=n_estimators,random_state=random_state)
model.fit(X_train,y_train.values.ravel())

# save model
model_filename = "model_RandomForestRegressor.pkl"
with open(model_filename, 'wb') as outfile:
    pickle.dump(model,outfile)
    
    
"""
# Ada Boost Regressor, MAE=194
from sklearn.ensemble import AdaBoostRegressor
n_estimators = 100
random_state = 42
model = AdaBoostRegressor(n_estimators=n_estimators,random_state=random_state)
model.fit(X_train,y_train.values.ravel())

# save model
model_filename = "model_AdaBoostRegressor.pkl"
with open(model_filename, 'wb') as outfile:
    pickle.dump(model,outfile)
""" 

"""
# Bagging Regressor, MAE=119
from sklearn.ensemble import BaggingRegressor
random_state = 42
model = BaggingRegressor(random_state=random_state)
model.fit(X_train,y_train.values.ravel())

# save model
model_filename = "model_BaggingRegressor.pkl"
with open(model_filename, 'wb') as outfile:
    pickle.dump(model,outfile)   
    """
"""
# Gradient Boosting Regressor, MAE=144
from sklearn.ensemble import GradientBoostingRegressor
random_state = 42
model = GradientBoostingRegressor(random_state=random_state)
model.fit(X_train,y_train.values.ravel())

# save model
model_filename = "model_GradientBoostingRegressor.pkl"
with open(model_filename, 'wb') as outfile:
    pickle.dump(model,outfile)   
"""

#%% Run Model on Test Dataset
y_pred = model.predict(X_test)
# y_pred = model.predict(vectorizer.transform(['Vertical load test/ 600 mm/ 97 T']))
# y_pred = model.predict(vectorizer.transform(['yellow']))



#%% Analyze Results
# MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
display('========== Mean Absolute Error ==========')
display(mae)
display('========== END ==========')




#%% Machine Learning Model for 'UNIT'
target_names = 'UNIT1'


# encode categorical target
from sklearn.preprocessing import LabelEncoder
lenc = LabelEncoder()
_y = df[target_names].values
lenc.fit(_y.ravel())
df[target_names] = lenc.transform(_y.ravel())
target_possibilities = df.UNIT2.unique() # fix this, not universal

# save label encoder
lenc_filename= "lenc_target.pkl"
with open(lenc_filename, 'wb') as outfile:
    pickle.dump(lenc,outfile)

#%% save dataframe
lenc_filename= "dataframe.pkl"
with open(lenc_filename, 'wb') as outfile:
    pickle.dump(df,outfile)
    
    

#%% Split data

from sklearn.model_selection import train_test_split
y = df[target_names]

random_state = 42
test_size = 0.0
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
X_train = X
y_train = y
X_test = X
y_test = y

#%% Machine Learning Model
from sklearn.ensemble import RandomForestClassifier

n_estimators = 100
random_state = 42
model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
model.fit(X_train,y_train.values.ravel())


# save model
model_filename = "model_RandomForestClassifier.pkl"
with open(model_filename, 'wb') as outfile:
    pickle.dump(model,outfile)
    



#%% Run Model on Test Dataset
y_pred = model.predict(X_test)





#%% Analyze Results
# Feature Importance
feature_importance = list(zip(X_train, model.feature_importances_))
display('========== Feature Importance ==========')
display(feature_importance)
display('========== END ==========')

# Accuracy Score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
display('========== Accuracy Score ==========')
display(accuracy)
display('========== END ==========')


# Confusion Matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)

display('========== Confusion Matrix ==========')
display(conf_matrix)
display('========== END ==========')



# Plot Confusion Matrix
import matplotlib.pyplot as plt
plt.figure(figsize = (10, 10))
cmap = plt.cm.Blues
plt.imshow(conf_matrix,cmap=cmap)
plt.grid(None)
plt.title('Units Confusion Matrix', size = 24)
plt.colorbar(aspect=5)
output_labels = lenc.inverse_transform(target_possibilities)
tick_marks = np.arange(len(output_labels))
plt.xticks(tick_marks,output_labels,rotation=30)
plt.yticks(tick_marks,output_labels)
for ii in range(len(output_labels)):
    for jj in range(len(output_labels)):
        if conf_matrix[ii,jj] > np.max(conf_matrix)/2:
            plt.text(ii,jj,conf_matrix[ii,jj],horizontalalignment="center",color="white")
        else:
            plt.text(ii,jj,conf_matrix[ii,jj],horizontalalignment="center")
plt.savefig('RF_ConfusionMatrix.png')

plt.show()

#%% Build SpellCheck Dicionary
#!pip install pyspellchecker
from spellchecker import SpellChecker

spell = SpellChecker(language='en', case_sensitive=True)
infile = 'ConstructionWordListForSpellChecker.txt'
spell.word_frequency.load_text_file(infile)
outfile = 'ConstructionDictionaryForSpellChecker_incl_English.gz'
spell.export(outfile, gzipped=True)



