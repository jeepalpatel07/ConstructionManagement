# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:19:52 2021

@author: Danylo
"""

#%reset -f



#%% Function takes in string and outputs estimated time to complete and units
import pickle
import numpy as np
from datetime import datetime,timedelta
import math
import copy as copy


from flask import Flask, request, jsonify
from flask_cors import CORS

import json
import pandas as pd

CM_Software= Flask(__name__)
CORS(CM_Software)

# load model
model_filename= "model_RandomForestRegressor.pkl"
with open(model_filename, 'rb') as infile:
    model_REG = pickle.load(infile)

# load vectorizer    
vectorizer_filename= "vectorizer_count.pkl"
with open(vectorizer_filename, 'rb') as infile:
    vectorizer = pickle.load(infile)
 
# load vectorized X    
X_filename= "vectorizer_count_X.pkl"
with open(X_filename, 'rb') as infile:
    X = pickle.load(infile)    
 
    
# do the same as above for predicting units
# load model
model_filename= "model_RandomForestClassifier.pkl"
with open(model_filename, 'rb') as infile:
    model_CLS = pickle.load(infile)

# load Label Encoder    
lenc_filename= "lenc_target.pkl"
with open(lenc_filename, 'rb') as infile:
    lenc = pickle.load(infile)
    
# load dataframe
dataframe_filename = "dataframe.pkl"    
with open(dataframe_filename, 'rb') as infile:
    df = pickle.load(infile)
_a = df.shape
df_numrows = _a[0]

# load exact text for spellchecker
corpus_text = []
corpus_filename = "ConstructionWordListForSpellChecker.txt"
with open(corpus_filename, 'r', encoding="utf-8") as infile:
    corpus_text = infile.read()


#%% CM_Predict_DPU
def CM_Predict_DPU(inputstring):
    
    # pedict rate value, days per unit
    pred_dpu = model_REG.predict(vectorizer.transform([inputstring]))
    
    return pred_dpu

#%% CM_Predict_Units
def CM_Predict_Units(inputstring):
    
    # pedict unit type
    _pred_unit = model_CLS.predict(vectorizer.transform([inputstring]))
    _pred_unit_str = lenc.inverse_transform(_pred_unit.ravel())
    pred_unit = " "
    pred_unit = ' '.join(map(str,_pred_unit_str))
       
    return pred_unit

#%% CM_SearchForMatch
def CM_SearchForMatch(inputstring):
    # search for string
    _a = vectorizer.transform([inputstring])
    _b = X.dot(_a.transpose())
    _stringmatches = np.where(_b.todense() == int(max(_b.todense())))
    stringmatches = _stringmatches[0]

    return stringmatches


#%% Days per Unit - Lowest From Search
def CM_SearchLow(inputstring):
    stringmatches = CM_SearchForMatch(inputstring)
    _a = df['COMPLETION RATE DAYS PER UNIT'].values
    search_min_dpu = min(_a[stringmatches])
    return search_min_dpu

#%% Days per Unit - Highest From Search
def CM_SearchHigh(inputstring):
    stringmatches = CM_SearchForMatch(inputstring)
    _a = df['COMPLETION RATE DAYS PER UNIT'].values
    search_max_dpu = max(_a[stringmatches])
    return search_max_dpu


#%% Spell Check input string
#only if doesn't appear in corpus exactly
from spellchecker import SpellChecker
custom_dictionary_infile = 'ConstructionDictionaryForSpellChecker_incl_English.gz'
spell = SpellChecker(local_dictionary=custom_dictionary_infile,case_sensitive=False)
def CM_SpellCheck(inputstring):
    splitstring = inputstring.split(' ')
    for ii in range(0,len(splitstring)):
        # check if word is in construction dictionary
        if not(splitstring[ii].lower() in corpus_text.lower()):
            # correct if no exact match
            splitstring[ii] = spell.correction(splitstring[ii])
    outputstring = ' '.join(splitstring)
    return outputstring

#%% CM_Predict
def CM_Predict(Quantity,Transformation,Description,StartDate,EndDate):
    # reformat inputs if empty values
    if Quantity == []: Quantity = ['','','','','']
    if Transformation ==[]: Transformation = ['','','','','']
    if Description == []: Description = ['','','','','']
    if StartDate == []: StartDate = ''
    if EndDate == []: EndDate = ''    
    
    nextStartDate = []
    output = []
    
    # Error handling StartDate after EndDate
    if StartDate != '' and EndDate != '':
        if  datetime.strptime(StartDate,'%Y-%m-%d') > datetime.strptime(EndDate,'%Y-%m-%d'):
            output.append('Error: Start Date is after End Date<br />')
            output.append('Please correct and resubmit<br />')
            return
   
    for i_trans in range(len(Transformation)):
        if nextStartDate != []:
            StartDate = nextStartDate

        # Error handing, missing info
        if Transformation[i_trans] == 'OTHER' and Description[i_trans] == '':
            if i_trans == 0:
                output.append('Error: Please add additional information<br />')
            else:
                output.append('== Activity {}: {} ==<br />'.format(int(i_trans+1),Transformation[i_trans]))
                output.append('Error: Please add additional information<br />')
            continue

        if Transformation[i_trans] == 'other' or Transformation[i_trans] == 'Other' or Transformation[i_trans] == 'OTHER':
            Transformation[i_trans] = ''

        # Spell check on inputstring
        _sp_Transformation = CM_SpellCheck(Transformation[i_trans])
        _sp_Description = CM_SpellCheck(Description[i_trans])
        inputstring = _sp_Transformation + ' ' + _sp_Description
          
        # Error handling, no results     
        if len(CM_SearchForMatch(inputstring)) == df_numrows:
            if Transformation[i_trans] == '': Transformation[i_trans] = 'OTHER'
            output.append('== Activity {}: {} ==<br />'.format(int(i_trans+1),Transformation[i_trans]))
            output.append('== Desc.: {} ==<br />'.format(_sp_Description))
            output.append('Found zero results for Activity {}<br />'.format(int(i_trans+1)))
            output.append('Please add/adjust information<br />')
            continue
        
        pred_dpu = CM_Predict_DPU(inputstring)
        search_min_dpu = CM_SearchLow(inputstring)
        search_max_dpu = CM_SearchHigh(inputstring)
        if pred_dpu < search_min_dpu:
            pred_dpu = search_min_dpu
        if pred_dpu > search_max_dpu:
            pred_dpu = search_max_dpu   
        
        # All Scenarios
        output.append('== Activity {}: {} ==<br />'.format(int(i_trans+1),Transformation[i_trans]))
        output.append('== Desc.: {} ==<br />'.format(_sp_Description))
        output.append('Predicted Completion Rate, Days per Unit: {:.3f}<br />'.format(float(pred_dpu)))
        output.append('(min: {:.3f}, max: {:.3f})<br />'.format(float(search_min_dpu),float(search_max_dpu)))
        output.append('Predicted Completion Rate, Units per Day: {:.3f}<br />'.format(1/float(pred_dpu)))
        output.append('(min: {:.3f}, max: {:.3f})<br />'.format(1/float(search_max_dpu),1/float(search_min_dpu)))
        pred_unit = CM_Predict_Units(inputstring)
        output.append('Units: {}<br />'.format(pred_unit))
        
        # Scenario 1 - Quantity yes, startdate yes, enddate yes
        if Quantity[i_trans] != '' and StartDate != '' and EndDate != '':
            output.append('Quantity: {}<br />'.format(Quantity[i_trans]))
            output.append('Start Date: {}<br />'.format(StartDate))
            output.append('End Date: {}<br />'.format(EndDate))
            
            days_avail = datetime.strptime(EndDate,'%Y-%m-%d') - datetime.strptime(StartDate,'%Y-%m-%d')
            pred_numdays = math.ceil(float(Quantity[i_trans])*pred_dpu)
            newEndDate = datetime.strptime(StartDate,'%Y-%m-%d') + timedelta(days=pred_numdays)
            nextStartDate = datetime.strftime(newEndDate,'%Y-%m-%d')
            if float(Quantity[i_trans]) < (1/float(pred_dpu))*(days_avail/timedelta(days=1)):
                output.append(' "Sufficient Time Allocated to Project" <br />')
                output.append('Predicted Activity End Date: {}<br />'.format(datetime.strftime(newEndDate,'%Y-%m-%d')))
            else:
                output.append(' "Insufficient Time Allocated to Activity" <br />')
                output.append('Predicted Activity End Date: {}<br />'.format(datetime.strftime(newEndDate,'%Y-%m-%d')))


        # Scenario 2 - Quantity yes, startdate yes, enddate no
        elif Quantity[i_trans] != '' and StartDate != '' and EndDate == '':
            output.append('Quantity: {}<br />'.format(Quantity[i_trans]))
            output.append('Start Date: {}<br />'.format(StartDate))
            pred_numdays = math.ceil(float(Quantity[i_trans])*pred_dpu)
            newEndDate = datetime.strptime(StartDate,'%Y-%m-%d') + timedelta(days=pred_numdays)
            nextStartDate = datetime.strftime(newEndDate,'%Y-%m-%d')
            output.append('Total Predicted Completion Time: {} days<br /><br />'.format(pred_numdays))
            output.append('Predicted End Date: {}<br />'.format(datetime.strftime(newEndDate,'%Y-%m-%d')))


        # Scenario 3 - Quantity yes, startdate no, enddate yes
        elif Quantity[i_trans] != '' and StartDate == '' and EndDate != '':
            output.append('Quantity: {}<br />'.format(Quantity[i_trans]))
            output.append('End Date: {}<br />'.format(EndDate))
            pred_numdays = math.ceil(float(Quantity[i_trans])*pred_dpu)
            newStartDate = datetime.strptime(EndDate,'%Y-%m-%d') - timedelta(days=pred_numdays)
            EndDate = newStartDate
            output.append('Total Predicted Completion Time: {} days<br /><br />'.format(pred_numdays))
            output.append('To Complete Activity on Time, Start By Predicted Start Date: {}<br />'.format(datetime.strftime(newStartDate,'%Y-%m-%d')))

        
        # Scenario 4 - Quantity yes, startdate no, enddate no
        elif Quantity[i_trans] != '' and StartDate == '' and EndDate == '':
            output.append('Quantity: {}<br />'.format(Quantity[i_trans]))
            pred_numdays = math.ceil(float(Quantity[i_trans])*pred_dpu)
            output.append('Total Predicted Completion Time: {} days<br /><br />'.format(pred_numdays))

            
        # Scenario 5 - Quantity no, startdate yes, enddate yes
        elif Quantity[i_trans] == '' and StartDate != '' and EndDate != '':
            output.append('Start Date: {}<br />'.format(StartDate))
            output.append('End Date: {}<br />'.format(EndDate))
            num_days_available = (datetime.strptime(EndDate,'%Y-%m-%d') - datetime.strptime(StartDate,'%Y-%m-%d'))/timedelta(days=1)
            pred_units_completed = num_days_available/pred_dpu
            output.append('Predicted Number of Units Completed in Timeframe: {:.3f}<br />'.format(float(pred_units_completed)))

        # Scenario 6,7,8 - Quantity no, startdate no OR enddate no
        else:
            output.append('(Add additional info for extra analysis)<br />')
        
        # All Scenarios again
        output.append('=============================================<br />')


    # ugly quick fix line breaks on webpage
    # correct way is to fix json to allow for line breaks
  #  linelength = 51 
   # for ii in range(len(output)):
    #    if len(output[ii]) < linelength:       
     #       for jj in range(len(output[ii]),linelength,2):
      #          output[ii] = output[ii] + '. '
       # elif len(output[ii]) < linelength*2:
        #    for jj in range(len(output[ii]),linelength*2,2):
         #       output[ii] = output[ii] + '. '
    outputnew = ''.join(output)
    
    return outputnew    

    
#Api calling 
@CM_Software.route("/", methods=['GET'])
def index():
    print("THIS END POINT IS CALLED")
    return "This is my WEB page"

@CM_Software.route("/predict", methods=['POST'])
def predict():

    data = request.get_json("data")
    print("Dipali--11")
    print(data)
    print("-------")
    
    print("******")
    var = data["data"]
    print(var[0])
    print(var[1])
    print(var[2])
    print(var[3])
    print(var[4])

    

    output = CM_Predict(var[0],var[1],var[2],var[3],var[4]
        )

    print(output)

    resp = {
        "resp":output,
        
    }
    return resp

if __name__=='__main__':
    CM_Software.run(port=5007, debug=True)

#%% Sample Inputs For Testing 

'''
Quantity = ['33','','77','22','65']
Transformation = ['fire','fire','form work','','constraction']
Description = ['sys','','plinth','','']
StartDate = '2022-07-22'
EndDate = ''
'''

'''
Quantity = ['','','','','']
Transformation = ['','','','','']
Description = ['','','','','']
StartDate = ''
EndDate = ''
'''

#%% Run Program

'''
#from CM_Software import CM_Predict
output = CM_Predict(Quantity,Transformation,Description,StartDate,EndDate)
print('\n'.join(output))
'''
