import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from numpy import array
import math
import numpy
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from keras.models import load_model

def load_view():
    st.markdown("<h1 style='text-align: Center; color: White; margin-top: -80px;'>Price Predictor</h1>", unsafe_allow_html=True)
    val = ['blackgram','greengram','redgram','sesamum']
    value = st.selectbox('What is Crop in You want to grow ?',(val))
    if st.button('Click to get data'):
        st.subheader(" : Best Price You May Get after 30 days : ")
        data,df3 = get_prid(value)
        st.dataframe(data=data, width=None, height=None)
        st.line_chart(df3)    


    






def get_prid(crop_name):
    
    def removeoutlier(df):
        l = len(df['modal_price'])
        s = sum(df['modal_price'])
        avg = s/l
        new_df = df[(df['modal_price'] < avg* 1.5) & (df['modal_price'] > .5)]
        upper_limit = df['modal_price'].mean() + 3*df['modal_price'].std()
        lower_limit = df['modal_price'].mean() - 3*df['modal_price'].std()
        df['modal_price'] = np.where(
        df['modal_price']>upper_limit,
        upper_limit,
        np.where(df['modal_price']<lower_limit,lower_limit,df['modal_price']))
        return new_df




    df = pd.read_csv(r'{}.csv'.format(crop_name))
    df = removeoutlier(df)



    df1= df.reset_index()['modal_price']
    if len(df1)>1000:
        df1 = df1[:-1000]


    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


    training_size=int(len(df1)*0.7)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0] 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return numpy.array(dataX), numpy.array(dataY)


    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)


    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


    model = load_model('pro.h5')


    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)


    math.sqrt(mean_squared_error(y_train,train_predict))


    math.sqrt(mean_squared_error(ytest,test_predict))


    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)


    look_back=100
    trainPredictPlot = numpy.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    testPredictPlot = numpy.empty_like(df1)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
    plt.figure(figsize=(12,6))
    plt.plot(scaler.inverse_transform(df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()



    x_input=test_data[len(test_data)-100:].reshape(1,-1)


    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()



    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
        
        if(len(temp_input)>100):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input,verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1



    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)


    plt.figure(figsize=(12,6))
    plt.plot(day_new,df1[len(df1)-100:])
    plt.plot(day_pred,lst_output)


    plt.figure(figsize=(12,6))
    df3=df1.tolist()
    df3.extend(lst_output)
    plt.plot(df3)
    plt.xlabel('time')
    plt.ylabel('price')


    df3 = scaler.inverse_transform(df3)


    return (df3[len(df3)-31:]),df3







