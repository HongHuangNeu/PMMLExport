# coding:utf-8
import sklearn, sklearn.externals.joblib, sklearn_pandas, sklearn2pmml
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
from sklearn2pmml import PMMLPipeline
from sklearn.datasets import load_iris
from sklearn import tree


#
from sklearn2pmml import make_pmml_pipeline,sklearn2pmml

import pandas as pd
pd.set_option('display.max_columns', None)

from sklearn.externals import joblib
obj=joblib.load("model.m")

pmml_pipeline=make_pmml_pipeline(obj, active_fields = ['sbp','tobacco','ldl','adiposity','famhist','typea','obesity','alcohol','age'], target_fields =['chd'])
sklearn2pmml(pmml_pipeline,"result.pmml", with_repr = True,debug=True)

