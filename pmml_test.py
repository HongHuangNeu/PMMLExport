# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
from sklearn2pmml import PMMLPipeline





from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn_pandas import DataFrameMapper

import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import *
heart_data = pd.read_csv("heart.csv")
print heart_data
#LabelBinarizer()
#用Mapper定义特征工程
mapper = DataFrameMapper([
    (['sbp'], MinMaxScaler()),
    (['tobacco'], MinMaxScaler()),
    ('ldl', None),
    ('adiposity', None),
    ('famhist',LabelEncoder() ),
    ('typea', None),
    ('obesity', None),
    ('alcohol', None),
(['age'], Imputer(missing_values= -1, strategy="mean"))
],
df_out=True)



#用pipeline定义使用的模型，特征工程等
pipeline = PMMLPipeline([
    ('mapper', mapper),
   ("classifier",RandomForestClassifier())
#("classifier",LogisticRegression())
])

vv=pipeline.fit(heart_data[heart_data.columns.difference(["chd"])] ,heart_data["chd"] )
print vv
print pipeline.predict(heart_data[heart_data.columns.difference(["chd"])])
from sklearn.externals import joblib

joblib.dump(pipeline,"model.m")

