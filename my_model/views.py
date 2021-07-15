from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
model = pickle.load(open('user_offer_attraction.pkl', 'rb'))
dataset= pd.read_csv('marketing_dataset.csv')
X=dataset.drop(["Response","Income","Total Spent","Total Purchase","Recency","Complain","Teenhome"], axis = 1)

y=dataset[['Response']]
y=np.array(y,dtype='int64').ravel()

encoder=ColumnTransformer([('encoder',OneHotEncoder(), [0,1,19])],remainder='passthrough')
X=encoder.fit_transform(X)

temp1=X[:, 1:5]
temp2=X[:, 6:13]
temp3=X[:, 14:]

X=np.concatenate((temp1,temp2,temp3), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA
pca=PCA(n_components=27)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)

def index(request):
    return render(request,'index.html')

def prediction_model(l1):
    pred = np.array(l1)
    pred = pred.reshape(1,20)
    X_ui = encoder.transform(pred)

    temp1 = X_ui[:, 1:5]
    temp2 = X_ui[:, 6:13]
    temp3 = X_ui[:, 14:]

    X_ui = np.concatenate((temp1, temp2, temp3), axis=1)

    scaled_data = sc.transform(X_ui)

    out = pca.transform(scaled_data)
    output = model.predict(out)

    print("Response", output)
    return output




def result(request):
    l1=[]
    Education=request.POST.get('Education')
    Marital_Status=request.POST.get('Marital_Status')
    Kidhome=int(request.POST.get('Kidhome'))
    MntWines=int(request.POST.get('MntWines'))
    MntFruits=int(request.POST.get('MntFruits'))
    MntMeatProducts=int(request.POST.get('MntMeatProducts'))
    MntFishProducts=int(request.POST.get('MntFishProducts'))
    MntSweetProducts=int(request.POST.get('MntSweetProducts'))
    MntGoldProds=int(request.POST.get('MntGoldProds'))
    NumDealsPurchases=int(request.POST.get('NumDealsPurchases'))
    NumWebPurchases=int(request.POST.get('NumWebPurchases'))
    NumCatalogPurchases=int(request.POST.get('NumCatalogPurchases'))
    NumStorePurchases=int(request.POST.get('NumStorePurchases'))
    NumWebVisitsMonth=int(request.POST.get('NumWebVisitsMonth'))
    AcceptedCmp3=int(request.POST.get('AcceptedCmp3'))
    AcceptedCmp4=int(request.POST.get('AcceptedCmp4'))
    AcceptedCmp5=int(request.POST.get('AcceptedCmp5'))
    AcceptedCmp1=int(request.POST.get('AcceptedCmp1'))
    AcceptedCmp2=int(request.POST.get('AcceptedCmp2'))
    Country=request.POST.get('Country')

    l2=[Education,Marital_Status,Kidhome,MntWines,MntFruits,MntMeatProducts,MntFishProducts,MntSweetProducts,
      MntGoldProds,NumDealsPurchases,NumWebPurchases,NumCatalogPurchases,NumStorePurchases,NumWebVisitsMonth,
      AcceptedCmp3,AcceptedCmp4,AcceptedCmp5,AcceptedCmp1,AcceptedCmp2,Country]
    for i in l2:
        l1.append(i)
    print(l1)
    output=prediction_model(l1)
    print(output)
    if output == [1]:
        prediction = "customer accepted the offer in the campaign"
    else:
        prediction = "customer rejected the offer in the campaign"

    print(prediction)
    passs = {'prediction_text': 'Model has Predicted', 'output': prediction}
    return render(request,'result.html',passs)