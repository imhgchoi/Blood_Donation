import pandas as pd
from pandas import DataFrame
import numpy as np

train = pd.read_csv("D:/rawDataFiles/bloodDonation_train.csv")
test = pd.read_csv("D:/rawDataFiles/bloodDonation_test.csv")



testID = test.iloc[:, 0]
train.columns = ['x','last','number','volume','first','march']
test.columns = ['x','last','number','volume','first']

train['span'] = train['first'] - train['last']
train['persistability'] = train['volume']/(train['span']+1)
train['amountPer'] = train['volume']/train['number']

test['span'] = test['first'] - test['last']
test['persistability'] = test['volume']/(test['span']+1)
test['amountPer'] = test['volume']/test['number']

columns = ['last','first','span','number','persistability','amountPer']
X = train[columns]
y = train['march']
test = test[columns]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = DataFrame(scaler.transform(X_train))
X_train.columns = columns
X_test = DataFrame(scaler.transform(X_test))
X_test.columns = columns



from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(
    hidden_layer_sizes=(100, 100, 100),
    activation='relu',
    alpha=0.005,
    max_iter=10000,
    random_state=None
)
MLP.fit(X_train,y_train)
print("$$ MLP $$")
print('train accuracy : ' + str(MLP.score(X_train,y_train)))
print('test accuracy : ' + str(MLP.score(X_test,y_test)))

from sklearn.linear_model import LogisticRegression
GLM = LogisticRegression()
GLM.fit(X_train,y_train)
print("$$ GLM $$")
print('train accuracy : ' + str(GLM.score(X_train,y_train)))
print('test accuracy : ' + str(GLM.score(X_test,y_test)))

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(
    n_estimators=1000,
    max_depth=4,
    random_state=None,
    n_jobs=-1
)
RF.fit(X_train,y_train)
print("$$ RF $$")
print('train accuracy : ' + str(RF.score(X_train,y_train)))
print('test accuracy : ' + str(RF.score(X_test,y_test)))

from sklearn.svm import SVC
SVM = SVC(
    kernel='linear'
)
SVM.fit(X_train,y_train)
print("$$ SVM $$")
print('train accuracy : ' + str(SVM.score(X_train,y_train)))
print('test accuracy : ' + str(SVM.score(X_test,y_test)))

test = DataFrame(scaler.transform(test))
print(RF.predict(test))
print(MLP.predict(test))
print(GLM.predict(test))

RFsubmission = pd.DataFrame({'': testID, 'Made Donation in March 2007': RF.predict_proba(test)[:,1]})
MLPsubmission = pd.DataFrame({'': testID, 'Made Donation in March 2007': MLP.predict_proba(test)[:,1]})
GLMsubmission = pd.DataFrame({'': testID, 'Made Donation in March 2007': GLM.predict_proba(test)[:,1]})
print(RFsubmission)
print(RFsubmission.dtypes)

RFsubmission.to_csv("C:/Users/Froilan/Desktop/Repository/drivendataResultCSV/BloodDonation_RF.csv",index=False)
MLPsubmission.to_csv("C:/Users/Froilan/Desktop/Repository/drivendataResultCSV/BloodDonation_MLP.csv",index=False)
GLMsubmission.to_csv("C:/Users/Froilan/Desktop/Repository/drivendataResultCSV/BloodDonation_GLM.csv",index=False)