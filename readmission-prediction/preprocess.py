import pandas as pd
import numpy as np
import os

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

mimic_data_path = "/home/littlefield/mimic-data/mimiciii/1.4/"
def get_mimic_dataset(table_name):
    try:
        file = table_name + ".csv"
        return pd.read_csv(mimic_data_path + file, low_memory=False)
    except FileNotFoundError:
        print("Unable to load data table", table_name, "from", mimic_data_path + file)
        
def preprocess(admissions, notes):
    # Convert dates
    admissions.ADMITTIME = pd.to_datetime(admissions.ADMITTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    admissions.DISCHTIME = pd.to_datetime(admissions.DISCHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    admissions.DEATHTIME = pd.to_datetime(admissions.DEATHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    
    # sort by subject_ID and admission date
    admissions = admissions.sort_values(['SUBJECT_ID','ADMITTIME'])
    admissions = admissions.reset_index(drop = True)
    
    # add the next admission date and type for each subject using groupby
    # you have to use groupby otherwise the dates will be from different subjects
    admissions['NEXT_ADMITTIME'] = admissions.groupby('SUBJECT_ID').ADMITTIME.shift(-1)

    # get the next admission type
    admissions['NEXT_ADMISSION_TYPE'] = admissions.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)
    
    # get rows where next admission is elective and replace with naT or nan
    rows = admissions.NEXT_ADMISSION_TYPE == 'ELECTIVE'
    admissions.loc[rows,'NEXT_ADMITTIME'] = pd.NaT
    admissions.loc[rows,'NEXT_ADMISSION_TYPE'] = np.NaN
    
    # sort by subject_ID and admission date
    # it is safer to sort right before the fill in case something changed the order above
    admissions = admissions.sort_values(['SUBJECT_ID','ADMITTIME'])

    # back fill (this will take a little while)
    admissions[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']] = admissions.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']].fillna(method = 'bfill')
    
    # calculate number of days till next admission
    admissions['DAYS_NEXT_ADMIT'] =  (admissions.NEXT_ADMITTIME - admissions.DISCHTIME).dt.total_seconds()/(24*60*60)
    
    # filter to use discharge notes only
    notes_dis = notes.loc[notes.CATEGORY == 'Discharge summary']

    notes_dis_last = (notes_dis.groupby(['SUBJECT_ID','HADM_ID']).nth(-1)).reset_index()
    assert notes_dis_last.duplicated(['HADM_ID']).sum() == 0, 'Multiple discharge summaries per admission'
    
    # merge admissions and notes
    adm_notes = pd.merge(admissions[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DAYS_NEXT_ADMIT','NEXT_ADMITTIME','ADMISSION_TYPE','DEATHTIME']],
                        notes_dis_last[['SUBJECT_ID','HADM_ID','TEXT']], 
                        on = ['SUBJECT_ID','HADM_ID'],
                        how = 'left')
    assert len(admissions) == len(adm_notes), 'Number of rows increased'
    
    # ignore NEWBORN admissions, information is incomplete, and likely stored in another database
    adm_notes_clean = adm_notes.loc[adm_notes.ADMISSION_TYPE != 'NEWBORN'].copy()
    
    # generate output label: readmitted within 30 days
    adm_notes_clean['OUTPUT_LABEL'] = (adm_notes_clean.DAYS_NEXT_ADMIT < 30).astype('int')

    # remove patients who died while admitted
    adm_notes_clean = adm_notes_clean[adm_notes_clean.DEATHTIME.isnull()]
    
    # shuffle the samples
    adm_notes_clean = adm_notes_clean.sample(n = len(adm_notes_clean), random_state = 42)
    adm_notes_clean = adm_notes_clean.reset_index(drop = True)
    
    print('Number of positive samples:', (adm_notes_clean.OUTPUT_LABEL == 1).sum())
    print('Number of negative samples:',  (adm_notes_clean.OUTPUT_LABEL == 0).sum())
    print('Total samples:', len(adm_notes_clean))
    
    return adm_notes_clean
    
def subsample(train, pos_class=1, random_state=42):
    # split the training data into positive and negative
    rows_pos = train.OUTPUT_LABEL == pos_class
    train_pos = train.loc[rows_pos]
    train_neg = train.loc[~rows_pos]

    # merge the balanced data
    whole_df = pd.concat([train_pos, train_neg.sample(n = len(train_pos), random_state = random_state)],axis = 0)

    # shuffle the order of training samples 
    train_sub = whole_df.sample(n = len(whole_df), random_state = random_state).reset_index(drop = True)
    
    return train_sub

def undersample(X, y, samp_rate=0.5, random_state=42):
    rus = RandomUnderSampler(random_state=42, sampling_strategy=samp_rate)
    X_res, y_res = rus.fit_resample(np.array(X).reshape(-1, 1), y)
    combined = pd.DataFrame( data = {"TEXT": X_res.squeeze(), "OUTPUT_LABEL": y_res})
    
    return combined.fillna("")

def oversample(X, y, samp_rate=0.5, random_state=42):
    ros = RandomOverSampler(random_state=42, sampling_strategy=samp_rate)
    X_res, y_res = ros.fit_resample(np.array(X).reshape(-1, 1), y)
    combined = pd.DataFrame( data = {"TEXT": X_res.squeeze(), "OUTPUT_LABEL": y_res})
    
    return combined.fillna("")

def under_over_sample(X, y, under_samp_rate=0.15, over_samp_rate=0.75, random_state=42):
    under = RandomUnderSampler(sampling_strategy=under_samp_rate, random_state=random_state, )
    over = RandomOverSampler(sampling_strategy=over_samp_rate, random_state=random_state)
    steps = [('under', under), ('over', over)]
    pipeline = Pipeline(steps = steps)
    
    X_res, y_res = pipeline.fit_resample(np.array(X).reshape(-1, 1), y)
    
    combined = pd.DataFrame( data = {"TEXT": X_res.squeeze(), "OUTPUT_LABEL": y_res})
    
    return combined.fillna("")

def train_valid_test_split(data, sampling_method, pct_splits=[0.7, 0.2, 0.1], random_state=42):
    # Save 30% of the data as validation and test data 
    valid_test=data.sample(frac=np.sum(pct_splits[1:]),random_state=random_state)

    valid = valid_test.sample(frac = 1 - pct_splits[1], random_state = random_state)
    test = valid_test.drop(valid.index)

    # use the rest of the data as training data
    train = data.drop(valid_test.index)
    
    if sampling_method == "SUBSAMPLE":
        train = subsample(train[["TEXT", "OUTPUT_LABEL"]])
    elif sampling_method == "UNDER" :
        train = undersample(train.TEXT, train.OUTPUT_LABEL)
    elif sampling_method == "OVER":
        train = oversample(train.TEXT, train.OUTPUT_LABEL)
    elif sampling_method == "UNDER_OVER":
        train = under_over_sample(train.TEXT, train.OUTPUT_LABEL)

    print('Train prevalence(n = %d):' %len(train), train.OUTPUT_LABEL.sum()/ len(train))
    print('Valid prevalence(n = %d):' %len(valid), valid.OUTPUT_LABEL.sum()/ len(valid))
    print('Test prevalence(n = %d):' %len(test), test.OUTPUT_LABEL.sum()/ len(test))
    
    return (train, valid, test)