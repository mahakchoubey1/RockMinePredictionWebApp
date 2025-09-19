import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)
def rockMine_prediction(input_data):

    #convert input data  into numpy and reshape it 
    input_2_numpyarray = np.asarray(input_data, dtype=float).reshape(1, -1)

    # prediction using the model
    prediction = model.predict(input_data_reshaped)

    if prediction[0]=='R':
        return ' 🪨 the object is rock'
    else:
        return " 💣 the object is mine"


    # Main  Streamlit UI Function to tell the user to enter the input
def main():

        st.title("Rock and Mine Prediction Web App")
        st.markdown("Enter the values for the **selected sonar features** below:")

 # Get only the selected features (not all 60)
        #selected_indices = selector.get_support(indices=True)
        #num_inputs = 60

      
        # Store inputs in a dictionary
       # inputs = {}
        #for i in range(1, num_inputs + 1):
            #inputs[f"C{i}"] = st.number_input(f"C{i}", min_value=0.0, format="%.4f")
        num_inputs = 60
        input_data = []
        for i in range(num_inputs):
            value = st.number_input(f"Feature {i+1}", value=0.0)
            input_data.append(value)

        if st.button('🔍 predict'):
           # inputforlist = [st.number_input(f"C{i}", min_value=0.0, format="%.4f") for i in range(1, num_inputs + 1)]

            Result=rockMine_prediction(input_data)
            st.success(Result)


if __name__ == "__main__":
    main()












