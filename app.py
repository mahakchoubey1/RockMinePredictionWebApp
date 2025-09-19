import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open("trained_model.pkl","rb") as f:
    model=pickle.load(f)

def rockMine_prediction(input_data):

    #convert input data into numpy
    input_2_numpyarray=np.asarray(input_data,dtype=float)
    input_data_reshaped = inputdata_2_numpyarray.reshaped(1,-1)# 1=dta contain 1 row and -1 tells the aray to take all the columns from the dataset


    # prediction = model.predict(input_data_reshaped)

    if prediction[0]=='R':
        return 'the object is rock'
    else:
        return "the object is mine"


    # Main Function to tell the user to enter the input
def main():

        st.title("Rock and Mine Prediction Web App")
        num_inputs = 60

        # Store inputs in a dictionary
        inputs = {}
        for i in range(1, num_inputs + 1):
            inputs[f"C{i}"] = st.number_input(f"C{i}", min_value=0.0, format="%.4f")

        if st.button('Test Results'):
            inputforlist = [st.number_input(f"C{i}", min_value=0.0, format="%.4f") for i in range(1, num_inputs + 1)]

            Result=rockMine_prediction(inputforlist)
            st.success(Result)


if __name__ == "__main__":
    main()












