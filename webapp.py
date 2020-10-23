#import libraries
import pandas as pd
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import streamlit as st;

#create a title and subtitle
st.write(""" 
# Water Quality Prediction
Detect if the water is fit for consumption using ML and python
""")

#Open and display an image
image = Image.open('/DSC_0036.jpg')
st.image(image, caption='ML', use_column_width=True)

#Get the data
df=pd.read_excel('/water_quality.xlsx')

#set a subheader
st.subheader('Data chart:')
chart = st.bar_chart(df)

#split data
feature_cols=['pH','Conductivity','Turbidity']

X=df[feature_cols]
y=df.Label

#finding the number of rows
n=len(y)

#train/test split
X_train=X.iloc[0:n-1,:]
y_train=y.iloc[0:n-1]
X_test=X.iloc[n-1:,:]
y_test=y.iloc[n-1:]

#get user input
def get_user_input():
    place = st.text_input("Location:")
    return place

place_name= get_user_input()
st.subheader('Location: ')
st.write(place_name)

#create and train the model
#SMOTE algorithm(balancing data)
sm=SMOTE(random_state=3,k_neighbors=5)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
X_train, y_train = shuffle( X_train_res, y_train_res, random_state=0)
 
#RandomForest classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

#Store the model predictions in a variable
y_predict=clf.predict(X_test)
st.write(y_predict)


