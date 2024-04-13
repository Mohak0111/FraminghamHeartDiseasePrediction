import pandas as pd
import numpy as np

def process_data_model_2(data):
    # Create a DataFrame with a single row using the provided data
    df = pd.DataFrame([data])

    # Apply the same preprocessing steps as during training
    df['enc_hr'] = df['heartRate'].apply(lambda x: 0 if x <= 60 else (1 if 60 < x <= 100 else 2))
    df['encode_age'] = df['age'].apply(lambda x: 0 if x <= 40 else (1 if 40 < x <= 55 else 2))

    df_copy = df.copy()
    df_copy['log_cigsPerDay'] = np.log1p(df_copy['cigsPerDay'])
    df_copy['log_totChol'] = np.log1p(df_copy['totChol'])
    df_copy['log_diaBP'] = np.log1p(df_copy['diaBP'])
    df_copy['log_BMI'] = np.log1p(df_copy['BMI'])
    df_copy['log_heartRate'] = np.log1p(df_copy['heartRate'])
    df_copy['log_glucose'] = np.log1p(df_copy['glucose'])
    df_copy['log_age'] = np.log1p(df_copy['age'])

    df_copy.drop(['cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'age', 'currentSmoker', 'education', 'enc_hr', 'encode_age'], axis=1, inplace=True)

    # Standardize the features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    cols = df_copy.columns

    norm_df = scaler.fit_transform(df_copy)
    norm_df = pd.DataFrame(data=norm_df, columns=cols, index=df_copy.index)  

    return norm_df

def get_ds_2_model():
    df=pd.read_csv(filepath_or_buffer="datasets/framingham.csv")
    def impute_median(data):
        return data.fillna(data.median())
    #median imputation
    df.glucose = df['glucose'].transform(impute_median)
    df.education = df['education'].transform(impute_median)
    df.heartRate = df['heartRate'].transform(impute_median)
    df.totChol = df['totChol'].transform(impute_median)
    df.BPMeds = df['BPMeds'].transform(impute_median)
    ## group by classes that are in relation with other classes
    by_currentSmoker = df.groupby(['currentSmoker'])
    df.cigsPerDay = by_currentSmoker['cigsPerDay'].transform(impute_median)
    by_age = df.groupby(['male','age'])
    df.BMI = by_age['BMI'].transform(impute_median)
    def encode_age(data):
        if data <= 40:
            return 0
        if data > 40 and data <=55:
            return 1
        else:
            return 2    

    #heart rate encoder
    def heartrate_enc(data):
        if data <= 60:
            return 0
        if data > 60 and data <=100:
            return 1
        else:
            return 2

    #applying functions
    df['enc_hr'] = df['heartRate'].apply(heartrate_enc)
    df['encode_age'] = df['age'].apply(lambda x : encode_age(x))
    df_copy = df.copy()
    df_copy['log_cigsPerDay'] = np.log1p(df_copy['cigsPerDay'])
    df_copy['log_totChol'] = np.log1p(df_copy['totChol'])
    #df_copy['log_sysBP'] = np.log1p(df_copy['sysBP'])
    df_copy['log_diaBP'] = np.log1p(df_copy['diaBP'])
    df_copy['log_BMI'] = np.log1p(df_copy['BMI'])
    df_copy['log_heartRate'] = np.log1p(df_copy['heartRate'])
    df_copy['log_glucose'] = np.log1p(df_copy['glucose'])
    df_copy['log_age'] = np.log1p(df_copy['age'])

    df_copy.drop(['cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'age','currentSmoker', 'education', 'enc_hr', 'encode_age'], axis=1, inplace=True)
    
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    cols = df_copy.drop(['TenYearCHD'], axis=1).columns

    norm_df = scaler.fit_transform(df_copy.drop(['TenYearCHD'], axis=1))
    norm_df = pd.DataFrame(data=norm_df, columns=cols, index=df_copy.drop(['TenYearCHD'], axis=1).index)  
    #train-test split
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc

    x = norm_df
    y = df_copy['TenYearCHD']

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=23)
    # train a logistic regression model on the training set
    from sklearn.linear_model import LogisticRegression

    log_reg_cw = LogisticRegression(solver='liblinear', class_weight='balanced')
    log_reg_cw.fit(x_train, y_train)
    #Applying SMOTE

    from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE

    smote = SMOTE(sampling_strategy='not majority')
    x_s_res, y_s_res = smote.fit_resample(x_train, y_train)
    est_reg = LogisticRegression(solver='liblinear', max_iter=1000, C=1).fit(x_s_res, y_s_res)
    # est_pred = est_reg.predict(x_test)
    return est_reg