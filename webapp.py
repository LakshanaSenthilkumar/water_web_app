#import libraries
import pandas as pd
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import streamlit as st
from firebase import firebase
from twilio.rest import Client

#firebase connectivity
firebase = firebase.FirebaseApplication("https://drought-d0g.firebaseio.com/",None)
result = firebase.get('/drought-d0g/water/','')

#creating a dataframe
new_df=pd.DataFrame.from_dict(result,orient='index')

#create a title and subtitle
st.write(""" 
# WATER STATISTICS
Analysis of water availability and fitness using IoT and ML
""")

#Open and display an image
image = Image.open('DSC_0036.jpg')
st.image(image, caption='The most precious resource', use_column_width=True)

#Get the data
df=pd.read_excel('water_quality.xlsx')

        
#get user input
def get_user_input():
    option = st.selectbox('Location:',('','Madurai', 'Chennai'))
    return option
    
place_name=get_user_input()

#data display
if place_name != "" :
    st.subheader('Location: ')
    st.write(place_name)
    #filtering dataset
    filtered = new_df[new_df['Location']==place_name]
    graph_df = filtered.set_index('Time')

    #set a subheader
    st.subheader('Water level and Conductivity chart:')
    g1=pd.DataFrame(graph_df,columns=['Water_level','Conductivity'])
    chart = st.area_chart(g1)

    g2=pd.DataFrame(graph_df,columns=['pH','Turbidity'])
    st.subheader('pH and Turbidity chart:')
    chart = st.line_chart(g2)

    #split data
    feature_cols=['pH','Conductivity','Turbidity']
    X= filtered[feature_cols]

    #finding the number of rows
    n=len(filtered)

    #train/test split
    X_train=df[feature_cols]
    y_train=df.Label
    X_test=X.iloc[n-1:,:]

    #create and train the model
    #SMOTE algorithm(balancing data)
    sm=SMOTE(random_state=3,k_neighbors=5)
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
    X_train, y_train = shuffle( X_train_res, y_train_res, random_state=0)
     
    #RandomForest classifier
    clf=RandomForestClassifier(n_estimators=100)

    #Train the model using the training sets
    clf.fit(X_train,y_train)

    st.subheader('Fitness of water for consumption:')
    #Store the model predictions in a variable
    y_predict=clf.predict(X_test)
    st.write(y_predict)

    st.subheader('Water Availability:')
    #water level
    if(filtered.iloc[0]['Water_level']>600):
        st.write('ALERT..Low water levels')
    else:
        st.write('You have sufficent water!!')

    #send sms using twilio
    account_sid = "ACfa6b1e47c5338f9ec1e1663bc95750a9"
    auth_token= "4c9e1ae537d28e5db10d4a3e8d80c0d8"
    
    st.subheader('To alert the Municipal corporation officials, please click the button below:')
    if(st.button('Alert SMS')):
        client= Client(account_sid,auth_token)
        client.messages.create(from_="+14159415889", body="WARNING!!..Water level is below threshold and the quality needs to be inspected",to="+919003433431")
        st.write('SMS sent successfully!!')
        

