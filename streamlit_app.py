import streamlit as st 
import pandas as pd
# import plotly.express as px 
# import matplotlib.pyplot as plt
# from st_aggrid import AgGrid

# import numpy as np # linear algebra
# import seaborn as sns
# import itertools
# import warnings
# import xgboost as xgb
# import lightgbm as lgb
# import os

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, Input
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# from collections import Counter
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer

# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.svm import SVR
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.neural_network import MLPRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

# from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, MinMaxScaler
# from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score, precision_score
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# from imblearn.over_sampling import SMOTE, RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
# from scipy.stats import randint

# import lazypredict
# from lazypredict.Supervised import LazyClassifier, LazyRegressor

# warnings.filterwarnings("ignore")

df = pd.read_csv('air_quality_health_impact_data/air_quality_health_impact_data.csv', sep=',')

def main() : 
#     # Membuat tab bar dengan dua tab
#     tabs = st.tabs(["Data", "Profile"])

#     with tabs[0]:
#         data()
#     with tabs[1]:
#         profile()
        
    # st.sidebar.title("Menu")
    # menu = st.sidebar.selectbox("Pilih halaman:", ["Home", "Profile"])

    # if menu == "Home":
    #     data()
    # elif menu == "Profile":
    #     profile()
# def load_data(file):
#     if file is not None:
#         return pd.read_csv(file)
#     return pd.DataFrame()
    
# def data():
#     st.title("Home")
#     st.write("Selamat datang di halaman Home!")

#     # set dataframe
# st.write('dataframe')
# st.dataframe(df)
    # Upload dataset
    # uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    # df = load_data(uploaded_file)

    # if df.empty:
    #     st.write("Silakan upload file CSV untuk melihat data.")
    #     return
        
#     st.write('Menampilkan Dataframe dengan St AgGrid')
#     AgGrid(df.sort_values('id', ascending=False).head(20))

#     col1, col2, col3 = st.columns(3)
#     with col1:
#         jml_row = len(df)
#         st.metric(label="Jumlah kolom", value=f"{jml_row} rows")
#     with col2:    
#         jml_col = len(df.columns)
#         st.metric(label="Jumlah row", value=f"{jml_col} rows")
#     with col3:    
#         st.metric(label="Temperature", value="70 °F", delta="1.2 °F")

#     col1, col2 = st.columns(2)
#     with col1:
#         st.write('bedrooms vs price')
#         fig,ax = plt.subplots()
#         plt.scatter(df['bedrooms'],df['price'])
#         st.pyplot(fig)
#     with col2:    
#         plotly_fig = px.scatter(df['bedrooms'],df['price'])
#         st.plotly_chart(plotly_fig)

#     click_me_btn = st.button('Click Me')
#     st.write(click_me_btn) #Return True kalo di Click 
#     check_btn = st.checkbox('Klik Jika Setuju')
#     if check_btn :
#         st.write('Anda Setuju')    
    
#     radio_button= st.radio('Choose below',[x for x in range(1,3)])
#     st.write('Anda Memilih',radio_button)
    
#     #slider 
#     age_slider = st.slider('Berapa Usia Anda',0,100)
#     st.write('Usia Anda',age_slider)
    
#     #Input (Typing)
#     num_input = st.number_input('Input Berapapun')
#     st.write('Kuadrat dari {} adalah {}'.format(num_input,num_input**2))

#     #sidebar 
#     sidebar_radio_button = st.sidebar.radio('Pilih Menu',options=['A','B','C'])
#     sidebar_checkbox = st.sidebar.checkbox('Checkbox di Sidebar')

#     #sidebar 
#     with st.form("Data Diri"):
#        st.write("Inside the form")
#        slider_val = st.slider('Berapa Usia Anda',0,100)
#        st.write('Anda Memilih',slider_val)
        
#        checkbox_val = st.checkbox('Klik Jika Setuju')
#        if check_btn :
#            st.write('Anda Setuju')    

#        # Every form must have a submit button.
#        submitted = st.form_submit_button("Submit")
#        if submitted:
#            st.write("slider", slider_val, "checkbox", checkbox_val)

#     st.write("Outside the form")
    
#     #columns :
#     col1, col2, col3 = st.columns(3)

#     with col1:
#        st.header("A cat")
#        st.image("https://static.streamlit.io/examples/cat.jpg")

#     with col2:
#        st.header("A dog")
#        st.image("https://static.streamlit.io/examples/dog.jpg")

#     with col3:
#        st.header("An owl")
#        st.image("https://static.streamlit.io/examples/owl.jpg")
#     #expander 
#     #dengan with atau dengan assignment 
#     expander = st.expander("Klik Untuk Detail ")
#     expander.write('Anda Telah Membuka Detail')

# def profile():
#     st.title("Data Overview")
#     st.write("Ini adalah halaman Profile.")
#     st.write("Di sini Anda dapat menambahkan konten untuk halaman Profile Anda.")
    # Tambahkan konten halaman Profile di sini
    

if __name__ == '__main__' : 
  main()
