import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import joblib
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Data Import
train_df=pd.read_csv('C:/Users/Cabinet/Documents/visual studio codes/Model deployment/survey lung cancer.csv')

  

train_df["GENDER"] = train_df["GENDER"].replace(['F'],'0')
train_df["GENDER"] = train_df["GENDER"].replace(['M'],'1')
train_df[["GENDER"]] = train_df[["GENDER"]].apply(pd.to_numeric, errors ='ignore')
train_df["LUNG_CANCER"]= train_df["LUNG_CANCER"].replace(["NO"],'0')
train_df["LUNG_CANCER"]= train_df["LUNG_CANCER"].replace(["YES"],'0')
train_df[["LUNG_CANCER"]] = train_df[["LUNG_CANCER"]].apply(pd.to_numeric, errors ='ignore')


#splitting Training and testing data
X = train_df.drop(columns=["LUNG_CANCER"])
y = train_df["LUNG_CANCER"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42 ) #test-train data split - 20/80

X_train.shape
X_test.shape


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



logreg = LogisticRegression(C=10)
logreg.fit(X_train, y_train)
Y_predict1 = logreg.predict(X_test)


filename='C:/Users/Cabinet/Documents/visual studio codes/Model deployment/LCP_model.pkl'
joblib.dump(logreg,filename)





import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import os


df=pd.read_csv('C:/Users/hp/Desktop/Project 1/survey lung cancer.csv', header=0 , index_col=None)

print(df.head())
print(df.columns)

df.reset_index()
le = LabelEncoder()
df1 = df.copy(deep=True)

df1.GENDER = le.fit_transform(df1.GENDER)
df1.LUNG_CANCER = le.fit_transform(df1.LUNG_CANCER)

scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
df2 = pd.DataFrame(scaler.fit_transform(df1),columns=df.columns,index=df.index)

X = df2.drop(columns=["LUNG_CANCER"],axis=1)
y = df2["LUNG_CANCER"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 50)


sm = SMOTE(random_state = 500)
X_res, y_res = sm.fit_resample(X_train, y_train)

model = XGBClassifier(learning_rate=0.2,n_estimators=5000,use_label_encoder=False,random_state=40)
model.fit(X_res, y_res)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)


filename='C:/Users/hp/Desktop/Project 1/LCP_lr_model.pkl'
joblib.dump(model,filename)