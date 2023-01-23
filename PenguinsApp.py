import numpy as np
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(layout="wide")
st.write("""
# Penguin Prediction App
This app predicts the **Palmer Penguin** species!
Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

upload_file = st.sidebar.file_uploader("Upload your input CSV file",type=["csv"])

if upload_file is not None:
    input_df = pd.read_csv(upload_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox("Sex",('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)',32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)

        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}

        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

# Combile user input with entire penguins dataset for encoding phase

penguin_df = pd.read_csv('penguins_cleaned.csv')
penguin_df = penguin_df.drop('species',axis=1)
df = pd.concat([input_df,penguin_df],axis=0)

# Encoding of ordinal features
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col],prefix = col)
    df = pd.concat([df,dummy],axis=1)
    del df[col]

# Select only the first row (the user input data)
df = df[:1]

st.subheader("User Input Features")

if upload_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

model = pickle.load(open('model.pkl','rb'))

prediction = model.predict(df)
prediction_proba = model.predict_proba(df)


st.subheader("Class label and their corresponding index number")
st.write(np.array(['Adelie','Chinstrap','Gentoo']))

st.subheader('Prediction')
penguin_species = ['Adelie','Chinstrap','Gentoo']
st.write(penguin_species[prediction[0]])

st.subheader('Prediction Probability')
st.write(prediction_proba)