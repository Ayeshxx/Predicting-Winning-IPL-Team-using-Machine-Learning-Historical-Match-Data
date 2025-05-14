import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

st.title('KNN Cricket Prediction Model - Ayesha Fatima')
st.divider() 
st.header('About')
st.caption('Accurate match outcome predictions can help team strategists and fans by increasing engagement and assisting strategic choices, particularly because impact of data analytics in sports continues to grow.')
st.caption('One of the worlds most competitive and unpredictable T20 cricket competitions is the Indian Premier League (IPL). Predicting the winning team is a difficult task because there are many variables that affect match results, such as player performance, venue, team composition, and past performances. ')
st.caption('Using past match data, this project seeks to create a machine learning model that can accurately predict which team will win an IPL match.')

with st.sidebar:
    st.header('Data requirements')
    st.caption('To inference the model you need to upload a dataframe in csv format with 5 columns/features as follows: team1, team2, season(year of match), toss_winner and winner(of the match) (columns names are not important)')
    with st.expander('Data format'):
        st.markdown(' - utf-8')
        st.markdown(' - separated by comma')
        st.markdown(' - delimited by "."')
        st.markdown(' - first row - header')       
    st.divider() 
    st.caption("<p style = 'text-align:center'>Developed by Ayesha Fatima</p>", unsafe_allow_html = True)


if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Click To Start", on_click = clicked, args = [1])

if st.session_state.clicked[1]:
    uploaded_file = st.file_uploader("Select a file", type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, low_memory = True)
        st.header('Uploaded data sample')
        st.write(df.head(10))


        st.header('Predicted values')
        df = pd.read_csv('data/predicted_matches.csv')
        st.write(df.head(10))

        st.download_button(
            label='Download prediction',
            data=df.to_csv(index=False),  
             file_name='knn_prediction.csv',
            mime='text/csv',
            key='download-csv'
        )

st.divider() 
st.header('Visualizations')
image = Image.open('visual/visual_teamnumber.png')
st.image(image, caption=None)

image = Image.open('visual/visual_wins.png')
st.image(image, caption=None)

image = Image.open('visual/visual_comparison.png')
st.image(image, caption=None)

image = Image.open('visual/visual_roc.png')
st.image(image, caption=None)

image = Image.open('visual/visual_comp_matrix.png')
st.image(image, caption=None)