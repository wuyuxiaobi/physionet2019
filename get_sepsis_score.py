#!/usr/bin/env python

import numpy as np
from sklearn.externals import joblib
from lightgbm import LGBMClassifier
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def get_sepsis_score(data, model):
    num_new_features=28
    values=data
    current_patient=pd.DataFrame(values)
    current_patient=current_patient.fillna(method='pad')
    current_record=np.array(current_patient)
    if len(current_record)==1:
        new_feature=np.full((1,num_new_features),np.nan) 
    else:

        SaO2FiO2_min=np.nan_to_num(np.nanmin(current_record[:,13]/current_record[:,10]))
        Platelets_min=np.nan_to_num(np.nanmin(current_record[:,33]))
        Bilirubin_direct_max=np.nan_to_num(np.nanmax(current_record[:,20]))
        Bilirubin_total_max=np.nan_to_num(np.nanmax(current_record[:,26]))
        MAP_min=np.nan_to_num(np.nanmin(current_record[:,4]))
        Creatinine_max=np.nan_to_num(np.nanmin(current_record[:,19]))

        HR_max=np.nan_to_num(np.nanmax(current_record[:,0]))
        BUN_max=np.nan_to_num(np.nanmax(current_record[:,15]))
        Resp_max=np.nan_to_num(np.nanmax(current_record[:,6]))
        Temp_max=np.nan_to_num(np.nanmax(current_record[:,2]))
        Calcium_min=np.nan_to_num(np.nanmin(current_record[:,17]))
        WBC_max=np.nan_to_num(np.nanmax(current_record[:,31]))
        Hct_min=np.nan_to_num(np.nanmin(current_record[:,28]))
        Hgb_min=np.nan_to_num(np.nanmin(current_record[:,29]))
        DBP_min=np.nan_to_num(np.nanmin(current_record[:,5]))
        Glucose_max=np.nan_to_num(np.nanmax(current_record[:,21]))
        SBP_min=np.nan_to_num(np.nanmin(current_record[:,3]))


        Fibrinogen_max=np.nan_to_num(np.nanmax(current_record[:,32]))
        PTT_max=np.nan_to_num(np.nanmax(current_record[:,30]))
        HCO3_min=np.nan_to_num(np.nanmin(current_record[:,9]))
        Magnesium_max=np.nan_to_num(np.nanmax(current_record[:,23]))
        AST_max=np.nan_to_num(np.nanmax(current_record[:,14]))
        Phosphate_max=np.nan_to_num(np.nanmax(current_record[:,24]))
        pH_max=np.nan_to_num(np.nanmax(current_record[:,11]))
        TroponinI_max=np.nan_to_num(np.nanmax(current_record[:,27]))
        O2Sat_max=np.nan_to_num(np.nanmax(current_record[:,1]))
        PaCO2_min=np.nan_to_num(np.nanmin(current_record[:,12]))
        Platelets_min=np.nan_to_num(np.nanmin(current_record[:,33]))
        new_feature=np.array([SaO2FiO2_min, Platelets_min, Bilirubin_direct_max,Bilirubin_total_max,MAP_min,Creatinine_max,HR_max,BUN_max,
            Resp_max,Temp_max,Calcium_min,WBC_max,Hct_min,Hgb_min,DBP_min,Glucose_max,SBP_min,Fibrinogen_max,PTT_max,HCO3_min,Magnesium_max,AST_max,
            Phosphate_max,pH_max,TroponinI_max,O2Sat_max,PaCO2_min,Platelets_min])
    new_feature=np.reshape(new_feature,(1,num_new_features))
    raw_feature=values[-1,:].reshape(1,-1)
    current_record=np.hstack((raw_feature,new_feature))
    scores=model.predict_proba(current_record)
    scores=scores[:,1] 
    labels=(scores>0.02)
    return (scores, labels)

def load_sepsis_model():
    xgb_model=joblib.load('./xgboost_newfeature.pkl')
    return xgb_model
