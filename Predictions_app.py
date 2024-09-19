import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import re
import pickle

df = pd.read_csv(r"D:/Copper_model_Project/Final_df.csv")

st.set_page_config(layout="wide")

st.title("Industrial Copper Modeling Predictions")

option = option_menu(None,options = ["Predict Status","Predict Selling Price"],
                       icons = ["check-circle-fill","coin"],
                       default_index=0,
                       orientation="horizontal", 
                       styles={"nav-link-selected": {"background-color": "#13d0f5"}})



country_options  = list(df['country'].unique())
application_options = list(df['application'].unique())
product_options = list(df['product_ref'].unique())
item_type_options = list(df['item type'].unique())
status_options = list(df['status'].unique())

if option == "Predict Status":

    with st.form('Classification'):

        col1, col2, col3 = st.columns([5, 2, 5])
        with col1:
            st.write(' ')
            customer = st.text_input('customer ID (Min: 12458, Max: 2147483647)')
            width = st.text_input('Enter width(Min: 1, Max: 2990)')
            quantity_tons = st.text_input('Quantity Tons (Min: 0.00001 & Max: 1000000000)')
            thickness = st.text_input('Thickness, (Min : 0.1, Max : 2500)')
            selling_price = st.text_input('Selling Price , (Min : 0.1, Max : 100001015)')
        
        with col3:
            st.write()
            country = st.selectbox('Country', sorted(country_options))
            application = st.selectbox('Application', sorted(application_options))
            product_ref = st.selectbox('Product', sorted(product_options))
            item_type = st.selectbox('Item Type', sorted(item_type_options))

            st.write('')
            st.write('')
            submit_button_class  = st.form_submit_button(label='SUBMIT')

    if submit_button_class :

        with open('class_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)

        with open('class_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('classification_model.pkl', 'rb') as f:
            class_model = pickle.load(f)

        
        new_data = np.array([[int(customer),country,application,float(width),product_ref, np.log(float(quantity_tons)), 
                              np.log(float(thickness)), np.log(float(selling_price)), item_type]])
        
        new_data_ohe = encoder.transform(new_data[:, [-1]])
        new_data = np.concatenate((new_data[:, [0, 1, 2, 3, 4, 5, 6, 7]], new_data_ohe), axis=1)
        new_data = scaler.transform(new_data)
        new_pred = class_model.predict(new_data)

        if new_pred == 1:
            st.write('## :green[The Predicted Status is : Won] ')
        else:
            st.write('## :red[The Predicted Status is : Lost] ')


elif option == "Predict Selling Price":

    with st.form('Regression'):
        col1, col2, col3 = st.columns([5, 2, 5])
        
        with col1:
            customer = st.text_input('customer ID (Min: 12458, Max: 2147483647)')
            width = st.text_input('Enter width(Min: 1, Max: 2990)')
            quantity_tons = st.text_input('Quantity Tons (Min: 0.00001 & Max: 1000000000)')
            thickness = st.text_input('Thickness, (Min : 0.1, Max : 2500)')
            status =  st.selectbox('Status', sorted(status_options)) 

        with col3:
            country = st.selectbox('Country', sorted(country_options))
            application = st.selectbox('Application', sorted(application_options))
            product_ref = st.selectbox('Product', sorted(product_options))
            item_type = st.selectbox('Item Type', sorted(item_type_options)) 

            st.write('')
            st.write('')
            submit_button_reg  = st.form_submit_button(label='SUBMIT')      
        
    if submit_button_reg :

        with open('reg_OHE_encoder.pkl', 'rb') as f:
            OHE_encoder = pickle.load(f)

        with open('reg_lb_encoder.pkl', 'rb') as f:
            lb_encoder = pickle.load(f)

        with open('reg_scaler.pkl', 'rb') as f:
            scaler_reg = pickle.load(f)

        with open('regression_model.pkl', 'rb') as f:
            reg_model = pickle.load(f)

        new_data_reg = np.array([[int(customer),country,application,float(width),product_ref, np.log(float(quantity_tons)), 
                              np.log(float(thickness)), item_type, status]])

        new_data_ohe1 = OHE_encoder.transform(new_data_reg[:, [-2]])
        new_data_lb = lb_encoder.transform(new_data_reg[:, [-1]]).reshape(-1, 1)
        new_data_reg = np.concatenate((new_data_reg[:, [0, 1, 2, 3, 4, 5, 6]], new_data_ohe1, new_data_lb), axis=1)
        new_data_reg = scaler_reg.transform(new_data_reg)
        new_pred_reg = reg_model.predict(new_data_reg)[0]

        st.write(f'## :green[Predicted selling price: {round(np.exp(new_pred_reg),2)}]')
