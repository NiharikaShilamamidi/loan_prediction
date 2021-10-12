import pandas as pd

import numpy as np

data=pd.read_csv("loan prediction.csv")


x=data.iloc[:,1:12]
y=data.iloc[:,12:]


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan , strategy='most_frequent')
imputer1=imputer.fit(x.iloc[:,:])
x.iloc[:,:]=imputer1.transform(x.iloc[:,:])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x["gender"] = LabelEncoder().fit_transform(x["Gender"].astype(str))
x["married"] = LabelEncoder().fit_transform(x["Married"].astype(str))
x["education"] = LabelEncoder().fit_transform(x["Education"].astype(str))
x["property_area"] = LabelEncoder().fit_transform(x["Property_Area"].astype(str))
x["self_employed"] = LabelEncoder().fit_transform(x["Self_Employed"].astype(str))


x.drop('Gender', inplace=True, axis=1)
x.drop('Education', inplace=True, axis=1)
x.drop('Property_Area', inplace=True, axis=1)
x.drop('Self_Employed', inplace=True, axis=1)
x.drop('Married', inplace=True, axis=1)

import matplotlib.pyplot as py

x.loc[x['Dependents'] =='3+', 'Dependents'] = 3

x['Dependents']=x['Dependents'].astype(float)

'''
py.hist(x.Dependents,bins=5,rwidth=0.5)
py.xlabel('Dependents')
py.ylabel('count')
py.show()


py.hist(x.ApplicantIncome,bins=5,rwidth=0.5)
py.xlabel('ApplicantIncome')
py.ylabel('count')
py.show()


py.hist(x.CoapplicantIncome ,bins=5,rwidth=0.5)
py.xlabel('CoapplicantIncome ')
py.ylabel('count')
py.show()


py.hist(x.LoanAmount,bins=5,rwidth=0.5)
py.xlabel('LoanAmount')
py.ylabel('count')
py.show()


py.hist(x.Loan_Amount_Term,bins=5,rwidth=0.5)
py.xlabel('Loan_Amount_Term')
py.ylabel('count')
py.show()



py.hist(x.Credit_History,bins=5,rwidth=0.5)
py.xlabel('Credit_History')
py.ylabel('count')
py.show()


py.hist(x.gender ,bins=5,rwidth=0.5)
py.xlabel('gender ')
py.ylabel('count')
py.show()



py.hist(x.married,bins=5,rwidth=0.5)
py.xlabel('married')
py.ylabel('count')
py.show()

py.hist(x.education,bins=5,rwidth=0.5)
py.xlabel('education')
py.ylabel('count')
py.show()


py.hist(x.property_area,bins=5,rwidth=0.5)
py.xlabel('property_area')
py.ylabel('count')
py.show()

py.hist(x.self_employed,bins=5,rwidth=0.5)
py.xlabel('self_employed')
py.ylabel('count')
py.show()
'''


IQR=x.Dependents.quantile(0.75)-x.Dependents.quantile(0.25)
lower_bridge=x['Dependents'].quantile(0.25)-(IQR*1.5)
upper_bridge=x['Dependents'].quantile(0.75)+(IQR*1.5)
x.loc[x['Dependents']>=2.5,'Dependents']=2.5
x.loc[x['Dependents']<-1.5,'Dependents']=-1.5




x['ApplicantIncome']=x['ApplicantIncome'].astype(float)
IQR=x.ApplicantIncome.quantile(0.75)-x.ApplicantIncome.quantile(0.25)
lower_bridge=x['ApplicantIncome'].quantile(0.25)-(IQR*1.5)
upper_bridge=x['ApplicantIncome'].quantile(0.75)+(IQR*1.5)
x.loc[x['ApplicantIncome']>=upper_bridge,'ApplicantIncome']=upper_bridge
x.loc[x['ApplicantIncome']<lower_bridge,'ApplicantIncome']=lower_bridge



x['CoapplicantIncome']=x['CoapplicantIncome'].astype(float)

IQR=x.CoapplicantIncome.quantile(0.75)-x.CoapplicantIncome.quantile(0.25)
lower_bridge=x['CoapplicantIncome'].quantile(0.25)-(IQR*1.5)
upper_bridge=x['CoapplicantIncome'].quantile(0.75)+(IQR*1.5)
x.loc[x['CoapplicantIncome']>=upper_bridge,'CoapplicantIncome']=upper_bridge
x.loc[x['CoapplicantIncome']<lower_bridge,'CoapplicantIncome']=lower_bridge


x['LoanAmount']=x['LoanAmount'].astype(float)

IQR=x.LoanAmount.quantile(0.75)-x.LoanAmount.quantile(0.25)
lower_bridge=x['LoanAmount'].quantile(0.25)-(IQR*1.5)
upper_bridge=x['LoanAmount'].quantile(0.75)+(IQR*1.5)
x.loc[x['LoanAmount']>=upper_bridge,'LoanAmount']=upper_bridge
x.loc[x['LoanAmount']<lower_bridge,'LoanAmount']=lower_bridge


x['Loan_Amount_Term']=x['Loan_Amount_Term'].astype(float)

IQR=x.Loan_Amount_Term.quantile(0.75)-x.Loan_Amount_Term.quantile(0.25)
lower_bridge=x['Loan_Amount_Term'].quantile(0.25)-(IQR*1.5)
upper_bridge=x['Loan_Amount_Term'].quantile(0.75)+(IQR*1.5)
x.loc[x['Loan_Amount_Term']>=upper_bridge,'Loan_Amount_Term']=upper_bridge
x.loc[x['Loan_Amount_Term']<lower_bridge,'Loan_Amount_Term']=lower_bridge


x['Credit_History']=x['Credit_History'].astype(float)

IQR=x.Credit_History.quantile(0.75)-x.Credit_History.quantile(0.25)
lower_bridge=x['Credit_History'].quantile(0.25)-(IQR*1.5)
upper_bridge=x['Credit_History'].quantile(0.75)+(IQR*1.5)
x.loc[x['Credit_History']>=upper_bridge,'Credit_History']=upper_bridge
x.loc[x['Credit_History']<lower_bridge,'Credit_History']=lower_bridge



x['gender']=x['gender'].astype(float)

IQR=x.gender.quantile(0.75)-x.gender.quantile(0.25)
lower_bridge=x['gender'].quantile(0.25)-(IQR*1.5)
upper_bridge=x['gender'].quantile(0.75)+(IQR*1.5)
x.loc[x['gender']>=upper_bridge,'gender']=upper_bridge
x.loc[x['gender']<lower_bridge,'gender']=lower_bridge


IQR=x.gender.quantile(0.75)-x.gender.quantile(0.25)
lower_bridge=x['gender'].quantile(0.25)-(IQR*1.5)
upper_bridge=x['gender'].quantile(0.75)+(IQR*1.5)
x.loc[x['gender']>=upper_bridge,'gender']=upper_bridge
x.loc[x['gender']<lower_bridge,'gender']=lower_bridge



IQR=x.gender.quantile(0.75)-x.gender.quantile(0.25)
lower_bridge=x['gender'].quantile(0.25)-(IQR*1.5)
upper_bridge=x['gender'].quantile(0.75)+(IQR*1.5)
x.loc[x['gender']>=upper_bridge,'gender']=upper_bridge
x.loc[x['gender']<lower_bridge,'gender']=lower_bridge



x['property_area']=x['property_area'].astype(float)

IQR=x.property_area.quantile(0.75)-x.property_area.quantile(0.25)
lower_bridge=x['property_area'].quantile(0.25)-(IQR*1.5)
upper_bridge=x['property_area'].quantile(0.75)+(IQR*1.5)
x.loc[x['property_area']>=upper_bridge,'property_area']=upper_bridge
x.loc[x['property_area']<lower_bridge,'property_area']=lower_bridge



x['self_employed']=x['self_employed'].astype(float)

IQR=x.self_employed.quantile(0.75)-x.self_employed.quantile(0.25)
lower_bridge=x['self_employed'].quantile(0.25)-(IQR*1.5)
upper_bridge=x['self_employed'].quantile(0.75)+(IQR*1.5)
x.loc[x['self_employed']>=upper_bridge,'self_employed']=upper_bridge
x.loc[x['self_employed']<lower_bridge,'self_employed']=lower_bridge

y["loan_status"] = LabelEncoder().fit_transform(y["Loan_Status"].astype(str))

y["loan_status"]=y['loan_status'].astype(float)

y.drop('Loan_Status', inplace=True, axis=1)

y=pd.DataFrame(y)
'''
y.value_counts().plot(kind='bar',figsize=(10,10))
'''
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)



import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X,y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
rfc = RandomForestClassifier(max_depth=2, random_state=0)
rfc.fit(x_train, np.ravel(y_train))
pred_rfc=rfc.predict(x_test)
print(accuracy_score(y_test, pred_rfc))

'''
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,np.ravel(y_train))
pred_lr=lr.predict(x_test)
accuracy_score(y_test, pred_lr)


from sklearn.naive_bayes import GaussianNB
gn=GaussianNB()
gn.fit(x_train,np.ravel(y_train))
pred_gn=gn.predict(x_test)
accuracy_score(np.ravel(y_test), pred_gn)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,np.ravel(y_train))
pred_knn=knn.predict(x_test)
accuracy_score(y_test, pred_knn)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
pred_dt=dt.predict(x_test)
accuracy_score(y_test, pred_dt)

from sklearn.svm import SVC
svc=SVC(kernel="linear",C=0.025,random_state=101)
svc.fit(x_train,np.ravel(y_train))
pred_svc=svc.predict(x_test)
accuracy_score(y_test, pred_svc)

from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier()
sgd.fit(x_train,np.ravel(y_train))
pred_sgd=sgd.predict(x_test)
accuracy_score(y_test, pred_sgd)
'''


import pickle
pickle.dump(rfc,open('model1.pkl','wb'))

mod=pickle.load(open('model1.pkl','rb'))

