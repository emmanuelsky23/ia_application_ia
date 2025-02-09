import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Titre de l'application
st.title("Prédiction des Maladies avec IA")

# Uploader le fichier CSV
uploaded_file = st.file_uploader("Téléchargez votre fichier Training.csv", type="csv")

if uploaded_file is not None:
    # Charger les données
    train_data = pd.read_csv(uploaded_file)

    # Préparer les données
    X = train_data.drop(['prognosis', 'Unnamed: 133'], axis=1)
    y = train_data['prognosis']

    # Entraîner le modèle
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, "modele_prediction.pkl")

    # Charger le modèle
    model = joblib.load("modele_prediction.pkl")

    # Entrée des symptômes
    symptoms = st.text_input("Entrez vos symptômes séparés par des virgules")

    if st.button("Prédire"):
        if symptoms:
            symptom_list = symptoms.split(",")
            input_data = pd.DataFrame(columns=X.columns)
            input_data.loc[0] = [1 if symptom.strip() in symptom_list else 0 for symptom in X.columns]
            prediction = model.predict(input_data)
            st.write(f"La maladie prédite est : {prediction[0]}")
        else:
            st.warning("Veuillez entrer au moins un symptôme.")

# Section pour afficher les contributeurs
st.subheader("Projet réalisé par :")
st.write("N’SA _ MBWELIMA _ EMMANUEL")
st.write("SIMA _ ASANSI _ EMMANUEL")
st.write("MOKO _ MBOTA _ NAZAIRE")
st.write("LONGRI _ ASENGATER _ KEREN")
st.write("EBONDO _ MULEKE _ CYNTHIA")