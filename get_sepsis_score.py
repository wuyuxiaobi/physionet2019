#!/usr/bin/env python

import numpy as np
from sklearn.externals import joblib
from lightgbm import LGBMClassifier
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def get_sepsis_score(data, model):
    num_new_features=56
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


        SaO2FiO2_mean=np.nan_to_num(np.nanmean(current_record[-24:,13]/current_record[-24:,10]))
        Platelets_mean=np.nan_to_num(np.nanmean(current_record[-24:,33]))
        Bilirubin_direct_mean=np.nan_to_num(np.nanmean(current_record[-24:,20]))
        Bilirubin_total_mean=np.nan_to_num(np.nanmean(current_record[-24:,26]))
        MAP_mean=np.nan_to_num(np.nanmean(current_record[-24:,4]))
        Creatinine_mean=np.nan_to_num(np.nanmean(current_record[-24:,19]))

        HR_mean=np.nan_to_num(np.nanmean(current_record[-24:,0]))
        BUN_mean=np.nan_to_num(np.nanmean(current_record[-24:,15]))
        Resp_mean=np.nan_to_num(np.nanmean(current_record[-24:,6]))
        Temp_mean=np.nan_to_num(np.nanmean(current_record[-24:,2]))
        Calcium_mean=np.nan_to_num(np.nanmean(current_record[-24:,17]))
        WBC_mean=np.nan_to_num(np.nanmean(current_record[-24:,31]))
        Hct_mean=np.nan_to_num(np.nanmean(current_record[-24:,28]))
        Hgb_mean=np.nan_to_num(np.nanmean(current_record[-24:,29]))
        DBP_mean=np.nan_to_num(np.nanmean(current_record[-24:,5]))
        Glucose_mean=np.nan_to_num(np.nanmean(current_record[-24:,21]))
        SBP_mean=np.nan_to_num(np.nanmean(current_record[-24:,3]))


        Fibrinogen_mean=np.nan_to_num(np.nanmean(current_record[-24:,32]))
        PTT_mean=np.nan_to_num(np.nanmean(current_record[-24:,30]))
        HCO3_mean=np.nan_to_num(np.nanmean(current_record[-24:,9]))
        Magnesium_mean=np.nan_to_num(np.nanmean(current_record[-24:,23]))
        AST_mean=np.nan_to_num(np.nanmean(current_record[-24:,14]))
        Phosphate_mean=np.nan_to_num(np.nanmean(current_record[-24:,24]))
        pH_mean=np.nan_to_num(np.nanmean(current_record[-24:,11]))
        TroponinI_mean=np.nan_to_num(np.nanmean(current_record[-24:,27]))
        O2Sat_mean=np.nan_to_num(np.nanmean(current_record[-24:,1]))
        PaCO2_mean=np.nan_to_num(np.nanmean(current_record[-24:,12]))
        Platelets_mean=np.nan_to_num(np.nanmean(current_record[-24:,33]))

        new_feature=np.array([SaO2FiO2_min, Platelets_min, Bilirubin_direct_max,Bilirubin_total_max,MAP_min,Creatinine_max,HR_max,BUN_max,
            Resp_max,Temp_max,Calcium_min,WBC_max,Hct_min,Hgb_min,DBP_min,Glucose_max,SBP_min,Fibrinogen_max,PTT_max,HCO3_min,Magnesium_max,AST_max,
            Phosphate_max,pH_max,TroponinI_max,O2Sat_max,PaCO2_min,Platelets_min,SaO2FiO2_mean,Platelets_mean,Bilirubin_direct_mean, Bilirubin_total_mean,
            MAP_mean,Creatinine_mean,HR_mean,BUN_mean,Resp_mean,Temp_mean,Calcium_mean,WBC_mean,Hct_mean,Hgb_mean,DBP_mean,Glucose_mean,SBP_mean,
            Fibrinogen_mean,PTT_mean,HCO3_mean,Magnesium_mean,AST_mean,Phosphate_mean,pH_mean,TroponinI_mean,O2Sat_mean,PaCO2_mean,Platelets_mean])

    new_feature=np.reshape(new_feature,(1,num_new_features))
    raw_feature=values[-1,:].reshape(1,-1)
    current_record=np.hstack((raw_feature,new_feature))
    scores=model.predict_proba(current_record)
    scores=scores[:,1] 
    labels=(scores>0.015)
    return (scores, labels)

def load_sepsis_model():
    xgb_model=joblib.load('./xgboost_newfeature.pkl')
    return xgb_model
