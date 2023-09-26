import streamlit as st

import pandas as pd                                                       
from sklearn import preprocessing                                        
from sklearn.neighbors import KNeighborsClassifier                        
import numpy as np   

def load_view(df):
    st.markdown("<h1 style='text-align: Center; color: White; margin-top: -80px;'>Crop Yield Finder</h1>", unsafe_allow_html=True)
    val = []
    for i in range(500):
        val.append(i)
    n = st.selectbox('What is Nitrogen in soil (mg/kg) ?',(val[:100]))
    p = st.selectbox('What is Phosperous in soil ? (mg/kg)',(val[:200]))
    k = st.selectbox('What is Potassium in soil ? (mg/kg)',(val[:300]))
    t = st.selectbox('What is Temprature ? (Degree Celcius)',(val[:50]))
    h = st.selectbox('What is Humidity in soil ? (%)',(val))
    ph = st.selectbox('What is Ph in soil ? ',(val[1:11]))
    rain = st.selectbox('What is Rainfall in village in MM/ ?',(val[:300]))
    if st.button('Click to get data'):
        st.subheader("Best Crop You can Grow on this is : "+get_res(df,n,p,k,t,h,ph,rain))
    
    




   

def get_res(df,n,p,k,t,h,ph,rain):
        
    le = preprocessing.LabelEncoder() 
    crop = le.fit_transform(list(df["CROP"])) 


    NITROGEN = list(df["NITROGEN"])       
    PHOSPHORUS = list(df["PHOSPHORUS"])
    POTASSIUM = list(df["POTASSIUM"])
    TEMPERATURE = list(df["TEMPERATURE"])    
    HUMIDITY = list(df["HUMIDITY"])
    PH = list(df["PH"])
    RAINFALL = list(df["RAINFALL"])  


    features = list(zip(NITROGEN, PHOSPHORUS, POTASSIUM, TEMPERATURE, HUMIDITY, PH, RAINFALL))                    
    features = np.array([NITROGEN, PHOSPHORUS, POTASSIUM, TEMPERATURE, HUMIDITY, PH, RAINFALL])          


    features = features.transpose()                                                                                
    print(features.shape)                                                                                         
    print(crop.shape)    
    model = KNeighborsClassifier(n_neighbors=3)                                                                 
    model.fit(features, crop) 


    nitro,phosp,pot,temp,hum,ph,rain = n,p,k,t,h,ph,rain
    values = [nitro,phosp,pot,temp,hum,ph,rain]


    nitrogen_content =         values[0]                                                                                                        
    phosphorus_content =       values[1]                                                                                                        
    potassium_content =        values[2]                                                                                                        
    temperature_content =      values[3]                                                                                                        
    humidity_content =         values[4]                                                                                                         
    ph_content =               values[5]                                                                                                        
    rainfall =                 values[6]                                                                                                        
    predict1 = np.array([nitrogen_content,phosphorus_content, potassium_content, temperature_content, humidity_content, ph_content, rainfall])  
    print(predict1)                                                                                                                             
    predict1 = predict1.reshape(1,-1)                                                                              
    print(predict1)                                                                                                
    predict1 = model.predict(predict1)                                                                              
    print(predict1)   


    crop_name = str()
    crops_names = ["Apple","Banana","Blackgram","Chickpea","Coconut","Coffee","Cotton","Grapes","Jute",
    "Kidneybeans","Lentil","Maize","Mango","Mothbeans","Mungbeans","Muskmelon","Orange","Papaya","Pigeonpeas","Pomegranate","Rice"]
    crop_name = crops_names[int(predict1)]



    if int(humidity_content) >=1 and int(humidity_content)<= 33 :
        humidity_level = 'low humid'
    elif int(humidity_content) >=34 and int(humidity_content) <= 66:
        humidity_level = 'medium humid'
    else:
        humidity_level = 'high humid'

    if int(temperature_content) >= 0 and int(temperature_content)<= 6: 
        temperature_level = 'cool'
    elif int(temperature_content) >=7 and int(temperature_content) <= 25:
        temperature_level = 'warm'
    else:
        temperature_level= 'hot' 

    if int(rainfall) >=1 and int(rainfall) <= 100: 
        rainfall_level = 'less'
    elif int(rainfall) >= 101 and int(rainfall) <=200:
        rainfall_level = 'moderate'
    elif int(rainfall) >=201:
        rainfall_level = 'heavy rain'

    if int(nitrogen_content) >= 1 and int(nitrogen_content) <= 50:      
        nitrogen_level = 'less'
    elif int(nitrogen_content) >=51 and int(nitrogen_content) <=100:
        nitrogen_level = 'not to less but also not to high'
    elif int(nitrogen_content) >=101:
        nitrogen_level = 'high'

    if int(phosphorus_content) >= 1 and int(phosphorus_content) <= 50: 
        phosphorus_level = 'less'
    elif int(phosphorus_content) >= 51 and int(phosphorus_content) <=100:
        phosphorus_level = 'not to less but also not to high'
    elif int(phosphorus_content) >=101:
        phosphorus_level = 'high'

    if int(potassium_content) >= 1 and int(potassium_content) <=50:   
        potassium_level = 'less'
    elif int(potassium_content) >= 51 and int(potassium_content) <= 100:
        potassium_level = 'not to less but also not to high'
    elif int(potassium_content) >=101:
        potassium_level = 'high'

    if float(ph_content) >=0 and float(ph_content) <=5:    
        phlevel = 'acidic' 
    elif float(ph_content) >= 6 and float(ph_content) <= 8:
        phlevel = 'neutral'
    elif float(ph_content) >= 9 and float(ph_content) <= 14:
        phlevel = 'alkaline'


    return crop_name                                        







