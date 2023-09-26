import streamlit as st
from PIL import Image

def load_view():

    st.markdown("<h1 style='text-align: Center; color: White; margin-top: -200px;'>About Us</h1>", unsafe_allow_html=True)
    


    st.header('Our Mission : ')
    mission = """
    Our mission is to design ML algorithm for getting the proper crop to grow for soil & weather condition his/her field is in and maximum yield farmer is going to get for that crop and another model to predict the future price for the crop he/she is growing and also suggest the crop to grow with respect to supply and demand so that the supply could meet the demand in the market.
    """
    st.subheader(mission)

    st.header('How to Use : ')
    mission = """
    Price Pridiction : Select Respective Crop and Hit "Click to get data.. You will get future 30 days price"
    """
    st.subheader(mission)
    


