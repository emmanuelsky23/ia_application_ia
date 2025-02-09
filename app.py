import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Charger les données
train_data = pd.read_csv(r"C:\Users\SKY\Documents\Training.csv")

# Préparer les données
X = train_data.drop(['prognosis', 'Unnamed: 133'], axis=1)
y = train_data['prognosis']

# Entraîner le modèle
model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, "modele_prediction.pkl")

# Charger le modèle
model = joblib.load("modele_prediction.pkl")

# Titre de l'application
st.title("Prédiction des Maladies avec IA")
symptoms = st.text_input("Entrez vos symptômes")

if st.button("Prédire"):
    if symptoms:
        symptom_list = symptoms.split(",")
        input_data = pd.DataFrame(columns=X.columns)
        input_data.loc[0] = [1 if symptom in symptom_list else 0 for symptom in X.columns]
        prediction = model.predict(input_data)
        st.write(f"La maladie prédite est : {prediction[0]}")
    else:
        st.warning("Veuillez entrer au moins un symptôme.")
