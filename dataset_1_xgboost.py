import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def process_data_model_1(data):
    # Create a DataFrame with a single row using the provided data
    df = pd.DataFrame([data])

    # Apply label encoding to 'famhist' column
    enc = LabelEncoder()
    df['famhist'] = enc.fit_transform(df['famhist'])

    # Normalize features
    for column in df.columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

    return df

def get_ds_1_model():
    data=pd.read_csv("datasets/CHDdata.csv")
    # print(data)
    enc=LabelEncoder()
    data['famhist']=enc.fit_transform(data['famhist'])
    data=data.drop_duplicates()
    cls_0=data[data['chd']==0]
    cls_1=data[data['chd']==1]
    cls_1=cls_1.sample(302,replace=True)
    data=pd.concat([cls_0,cls_1],axis=0)
    for column in data.columns:
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    y=data['chd']
    x=data.drop(['chd'],axis=1)
    xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,stratify=y)
    model= XGBClassifier()
    model.fit(xtrain,ytrain)
    return model