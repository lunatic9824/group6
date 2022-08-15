
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection  import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib



data_group6 = pd.read_csv('F:\Assignments\sem2\sl\KSI.csv')
pd.set_option('display.max_columns', 20)


print(data_group6.info())
print(data_group6.shape)  
print(data_group6.describe())
print(data_group6.columns)

###Data exploration & data Cleaning
# Investigation of columns whose data type is object
for c in data_group6.columns:
    if data_group6[c].dtype=='object':
        
        print(c)
        print(data_group6[c].value_counts())
        print()


#the target column
print(data_group6['ACCLASS'].value_counts()) # target column

data_group6['ACCLASS'] = np.where(data_group6['ACCLASS'] == 'Non-Fatal Injury', 0, data_group6['ACCLASS'])
data_group6['ACCLASS'] = np.where(data_group6['ACCLASS'] == 'Property Damage Only', 0, data_group6['ACCLASS'])
data_group6['ACCLASS'] = np.where(data_group6['ACCLASS'] == 'Fatal', 1, data_group6['ACCLASS'])
data_group6['ACCLASS']=data_group6['ACCLASS'].astype('int64')
data_group6['ACCLASS'].dtype

#Transform the columns with binary data

binary_columns = ['PEDESTRIAN','CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','TRSN_CITY_VEH'\
 ,'EMERG_VEH','PASSENGER','SPEEDING','AG_DRIV','REDLIGHT','ALCOHOL','DISABILITY']


data_group6[binary_columns]= np.where(data_group6[binary_columns]=='<Null>',0,data_group6[binary_columns])
data_group6[binary_columns]= np.where(data_group6[binary_columns]=='Yes',1,data_group6[binary_columns])
data_group6[binary_columns].astype('int64')
data_group6[binary_columns]


# Percentage of the number of null data of each column

for c in data_group6.columns:
    
    print(c)
    print(data_group6[c][data_group6[c]=='<Null>'].count()/data_group6[c].count() * 100)
    print()

# Find columns which have too many null data(40%) - the columns should be dropped

drop_columns = []
for c in data_group6.columns:
    if data_group6[c][data_group6[c]=='<Null>'].count()/data_group6[c].\
        count() * 100 > 40:
            drop_columns.append(c)
            
# Drop columns with unique values such as ID

drop_columns = drop_columns + ['X','Y', 'INDEX_', 'ObjectId']
drop_columns
            
# drop unnecessary columns

data_group6 = data_group6.drop(drop_columns, axis=1)
data_group6.shape

#Replacing <Null> to NA

data_group6 = data_group6.replace('<Null>', np.nan, regex=False)




#Plot

###TIME (Year, Month, Hour) VS fatality 

#YEAR
sns.catplot(y='YEAR', kind='count', data=data_group6, hue='ACCLASS',orient="h")
    
#MONTH
data_group6['MONTH']=pd.DatetimeIndex(data_group6['DATE']).month
data_group6.drop(['DATE','TIME'], axis=1,inplace=True)
sns.catplot(y='MONTH', kind='count', data=data_group6, hue='ACCLASS',orient="h")

#HOUR
sns.catplot(y='HOUR', kind='count', data=data_group6, hue='ACCLASS',orient="h")
    

#Location

data_group6.plot(kind="scatter", x="LONGITUDE", y="LATITUDE",figsize=(10,7))


cat_columns=['ROAD_CLASS','DISTRICT','LOCCOORD','TRAFFCTL','VISIBILITY','LIGHT',\
 'RDSFCOND','IMPACTYPE','INVTYPE','INJURY','INVAGE','INITDIR','VEHTYPE'\
     ,'POLICE_DIVISION','HOOD_ID']



##fig, axes = plt.subplots(2,2)

##z=0
##for i in range(2):
##    for j in range(2): 
##        if z >= 15:
##            break
##       subplot_data = data_group6[data_group6['ACCLASS'] ==1]
##        subplot_data = subplot_data.groupby(data_group6[investigation[z]]).count()        
##        subplot_data['ACCLASS'].plot(ax=axes[i,j],kind='bar')
##        z+=1
        
        
    
#Each column vs fatality

for col in cat_columns:
    
    sns.catplot(y=col, kind='count', data=data_group6, hue='ACCLASS',orient="h")
    


#binary data columns

## Causes

causes_fatal = data_group6.pivot_table(index='ACCLASS', margins=False ,\
                                       values=['ALCOHOL', 'AG_DRIV','SPEEDING'\
                                               ,'REDLIGHT','DISABILITY'],\
                                           aggfunc=np.sum)
  
ratio_causes_fatal = causes_fatal.iloc[1]/causes_fatal.iloc[0]

ratio_causes_fatal.plot(figsize=(10,8), title="Non-fatal / Fatal VS auses", grid=True,kind='bar')
plt.ylabel('Ratio')


print(data_group6.isna().sum()/len(data_group6)*100)

data_group6_cleaned = data_group6[['HOUR','VISIBILITY', 'LIGHT', 'ROAD_CLASS','RDSFCOND','TRAFFCTL','ACCLASS']]


print(data_group6_cleaned.isna().sum()/len(data_group6_cleaned)*100)

data_group6_cleaned=data_group6_cleaned.dropna()

data_group6_cleaned= pd.get_dummies(data_group6_cleaned,drop_first=True)

data_group6_cleaned.info()

target_data = data_group6_cleaned['ACCLASS']
features_data = data_group6_cleaned.drop('ACCLASS',axis=1)

print(features_data.describe())
print(target_data.describe())
X_train, X_test, y_train, y_test = train_test_split(features_data, target_data, test_size=0.2, random_state=32,stratify=target_data)

data_group6_cleaned["LIGHT"].unique()



###### BOHYUN KIM - 301131832 - SVM

##Modeling 
pipe_svm_group6 = Pipeline([
    ('svc', SVC(random_state=32))
    ])


param_grid = [{
    'svc__kernel': ['rbf'],
    'svc__C':  [0.1, 1, 10,100],
    'svc__gamma': [1.0,3.0],
    'svc__degree':[2]}]


grid_search_group6 = GridSearchCV( estimator=pipe_svm_group6,
                                   param_grid = param_grid,
                                   scoring='accuracy',
                                   refit=True,
                                   verbose=3 )

grid_search_group6.fit(X_train, y_train)

print('Best parameters: ', grid_search_group6.best_params_)
print('Best estimator: ', grid_search_group6.best_estimator_)


best_model_group6 = grid_search_group6.best_estimator_
y_pred = best_model_group6.predict(X_test)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)



####### NIYANTA - 301224927  - Logistic Regression

#Before Tuning

from sklearn.linear_model import LogisticRegression
LR_clf= LogisticRegression(solver='lbfgs', max_iter=1000,random_state=27)
LR_clf.fit(X_train, y_train)
y_predict = LR_clf.predict(X_test)

#accuracy of the training dataset
print("Accuracy on the model : ",accuracy_score(y_test, y_pred))

#confusion mertics
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, precision_score, recall_score
print(confusion_matrix(y_test, y_predict))

#classification report
print("classification Report : " , classification_report(y_test,y_predict))
print("Precision", precision_score(y_test, y_predict))
print("Recall", recall_score(y_test, y_predict))

#After Tuning

from sklearn.model_selection import GridSearchCV
grid = {'penalty': ['l1', 'l2'], 
               'C':[0.01,0.1, 1],
               'solver':['saga'],
               'max_iter':[10000]
               }
grid_search = GridSearchCV(LR_clf, 
                           param_grid =grid,
                           scoring = 'accuracy',
                           refit=True,
                           verbose=3)
grid_search.fit(X_train, y_train)

#Best parameters
print('Best parameter : ',grid_search.best_params_)
print('Best estimator : ',grid_search.best_estimator_)
cvres = grid_search.cv_results_
best_log_model = grid_search.best_estimator_
best_log_model.predict(X_test)


print("After Tuning :")
print("Accuracy on the model : ", accuracy_score(y_test, y_predict))
print("Precision : ", precision_score(y_test, y_predict))
print("Recall: ", recall_score(y_test, y_predict))
print(classification_report(y_test, y_predict))



######## PANKAJ SHARMA -             - Neural Network

MLP_Classifier= MLPClassifier(random_state=27)

parameter_space = {
    'hidden_layer_sizes': [3,(8,3),(15, 8, 2)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'lbfgs','adam'],
    'alpha': [1e-3,0.0001],
    'learning_rate': ['constant','adaptive'],
}

MLP_group6 = GridSearchCV(estimator=MLP_Classifier,
                            param_grid=parameter_space,
                            cv=5,
                                   scoring='accuracy',
                                   refit=True,
                                   verbose=3 )


MLP_group6.fit(X_train, y_train)
print('Best parameter(MLP) : ',MLP_group6.best_params_)
print('Best estimator(MLP) : ',MLP_group6.best_estimator_)

cvres = MLP_group6.cv_results_
best_log_model = MLP_group6.best_estimator_
best_log_model.predict(X_test)
y_predict = MLP_group6.predict(X_test)
print("Accuracy on the MLP model : ",accuracy_score(y_test, y_predict))
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, precision_score, recall_score
print(confusion_matrix(y_test, y_predict))


print("Accuracy on the model : ", accuracy_score(y_test, y_predict))
print("Precision : ", precision_score(y_test, y_predict))
print("Recall: ", recall_score(y_test, y_predict))
print(classification_report(y_test, y_predict))




######## Ambuj Mittal  - 301215925   - Random Forest
pipe_rf_group6 = Pipeline([
    ('rf', RandomForestClassifier(random_state=42))
])

params_grid = {
    'rf__n_estimators': [120, 140, 160],
    'rf__max_depth': [3, 5, 8],
    'rf__min_samples_split': [2, 3, 4],
    'rf__min_samples_leaf': [1, 3, 5], 
    'rf__class_weight': [ {0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 3}, 'balanced']
    }

grid_search_group6 = GridSearchCV( estimator=pipe_rf_group6,
                                   param_grid = params_grid,
                                   scoring='accuracy',
                                   refit=True,
                                   verbose=3 )

grid_search_group6.fit(X_train, y_train)

print('Best parameters: ', grid_search_group6.best_params_)
print('Best estimator: ', grid_search_group6.best_estimator_)


best_model_group6 = grid_search_group6.best_estimator_
y_pred = best_model_group6.predict(X_test)


from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, precision_score, recall_score
print(confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print(classification_report(y_test, y_pred))





######## Josewin       - 301224633            - Decision Tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=3, criterion = 'entropy', random_state=42)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Validate using 10 cross fold
print(" 10-fold cross-validation ")
clf = DecisionTreeClassifier(min_samples_split=20,criterion = 'entropy',
                                random_state=42)
clf.fit(X_train, y_train)
scores= cross_val_score(\
   clf, X_train, y_train, cv=10, scoring='f1_macro')


#Predict using the test set
y_pred = clf.predict(X_test)
#Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
"""
fine tune the model using grid search

"""
# set of parameters to test
param_grid = {
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }
print("-- Grid Parameter Search via 10-fold CV")
dt = DecisionTreeClassifier(criterion = 'entropy')
grid_search = GridSearchCV(dt,
                               param_grid=param_grid,
                               cv=10)

grid_search.fit(X_train, y_train)



print('Best Parameters are:',grid_search.best_params_)

#Predict the response for test dataset using the best parameters
dt = DecisionTreeClassifier(max_depth =5,min_samples_split= 2, criterion = 'entropy', min_samples_leaf= 10, random_state=42 )
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)

neww=[[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]



neww1=[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]

y_pred = dt.predict(neww1)

print(y_pred)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


joblib.dump(dt,"decisiontree.pkl")

#save model
import pickle
filename = 'DecisionTree_final_1.pickle'
pickle.dump(dt, open(filename, 'wb'))


