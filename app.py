import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")
df = df.dropna()

# Select required columns
df = df[['Genre', 'Director', 'Actor 1', 'Rating']]

# Encode categorical columns
genre_encoder = LabelEncoder()
director_encoder = LabelEncoder()
actor_encoder = LabelEncoder()

df['Genre'] = genre_encoder.fit_transform(df['Genre'])
df['Director'] = director_encoder.fit_transform(df['Director'])
df['Actor 1'] = actor_encoder.fit_transform(df['Actor 1'])

# Features and target
X = df[['Genre', 'Director', 'Actor 1']]
y = df['Rating']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# App UI
st.title("🎬 Movie Rating Prediction")

st.write("Predict movie ratings based on Genre, Director, and Actor.")

genre = st.selectbox("Select Genre", genre_encoder.classes_)
director = st.selectbox("Select Director", director_encoder.classes_)
actor = st.selectbox("Select Actor", actor_encoder.classes_)

if st.button("Predict Rating"):
    genre_encoded = genre_encoder.transform([genre])[0]
    director_encoded = director_encoder.transform([director])[0]
    actor_encoded = actor_encoder.transform([actor])[0]

    prediction = model.predict(
        [[genre_encoded, director_encoded, actor_encoded]]
    )

    st.success(f"Predicted Movie Rating: ⭐ {prediction[0]:.2f}")