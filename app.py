from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import streamlit as st
import numpy as np
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Load CSV data
@st.cache_data  # Cache to avoid reloading on every interaction
def load_data():
    return pd.read_csv("diabetes.csv")  # Replace with your file path
  
# Modeli yükle
@st.cache_resource
def load_bayes_pca_model():
    return joblib.load("bayes_pca_model.pkl")


@st.cache_resource
def load_lr_model():
    return joblib.load("lr_model.pkl")
  
@st.cache_resource
def load_best_lr_smote_model():
    return joblib.load("best_lr_smote_model.pkl")
  
@st.cache_resource
def load_lda():
    return joblib.load("lda.pkl")
  
@st.cache_resource
def load_pca():
    return joblib.load("pca.pkl")
  
@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

df = load_data()
data = pd.DataFrame(df)

gb = GridOptionsBuilder.from_dataframe(data)
gb.configure_selection("single")  # Allow single row selection
gb.configure_default_column(
    width=80,
    wrapHeaderText=True,  # Başlık metnini kaydır
    autoHeaderHeight=True  # Başlık yüksekliğini otomatik ayarla
)

grid_options = gb.build()

grid_response = AgGrid(
    data,
    gridOptions=grid_options,
    enable_enterprise_modules=False,
    height=300,
    key="sidebar_table",
)



col1, col2 = st.columns(2)

selected_rows = grid_response["selected_rows"]
# Check selected rows
if selected_rows is not None and not selected_rows.empty:
  with col1:
    pregnancies = st.number_input("Pregnancies", value=selected_rows.iloc[0]["Pregnancies"])  
    glucose = st.number_input("Glucose", value=selected_rows.iloc[0]["Glucose"])  
    blood_pressure = st.number_input("Blood Pressure", value=selected_rows.iloc[0]["BloodPressure"])  
    skin_thickness = st.number_input("Skin Thickness(mm)", value=selected_rows.iloc[0]["SkinThickness"]) 
  with col2:
    insulin = st.number_input("Insulin(mu U/ml)", value=selected_rows.iloc[0]["Insulin"]) 
    bmi = st.number_input("BMI", value=selected_rows.iloc[0]["BMI"])  
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", value=selected_rows.iloc[0]["DiabetesPedigreeFunction"])  
    age = st.number_input("Age", value=selected_rows.iloc[0]["Age"])  
  with st.sidebar:
      st.header("RESULTS")
else:
  with col1:
    pregnancies = st.number_input("Pregnancies", value=0)
    glucose = st.number_input("Glucose", value=0)
    blood_pressure = st.number_input("Blood Pressure", value=0)
    skin_thickness = st.number_input("Skin Thickness(mm)", value=0)
  with col2:
    insulin = st.number_input("Insulin(mu U/ml)", value=0)
    bmi = st.number_input("BMI", value=0)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", value=0)
    age = st.number_input("Age", value=0)
  with st.sidebar:
      st.header("RESULTS")
# Load the model
bayes_pca_model = load_bayes_pca_model()
lr_model = load_lr_model()
best_lr_smote_model = load_best_lr_smote_model()
lda = load_lda()
pca = load_pca()
scaler = load_scaler()

features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]

if st.button("Predict"):

    input_array = np.array(features).reshape(1, -1) # Shape: 1, 8
    input_scaled = scaler.transform(input_array) 
    input_pca = pca.transform(input_scaled)
    
    bayes_pca_prediction = bayes_pca_model.predict(input_pca)
    
    with st.sidebar:
      st.header("RESULTS")
      # st.success(f"Bayes PCA Tahmin: {bayes_pca_prediction[0]}")
      bayes_pca_prediction_proba = bayes_pca_model.predict_proba(input_pca)
      
      lr_prediction = lr_model.predict(input_scaled)
      # st.success(f"Logistic Regression Tahmin: {lr_prediction[0]}")
      lr_prediction_proba = lr_model.predict_proba(input_scaled)
      
      best_prediction = best_lr_smote_model.predict(input_array)
      # st.success(f"Logistic Regression SMOTE Tahmin: {best_prediction[0]}")
      best_prediction_proba = best_lr_smote_model.predict_proba(input_array)
      
      
      # Sonuçları gösterme
      st.subheader("Bayes PCA Tahmin Sonucu")
      if bayes_pca_prediction[0] == 1:
          st.error(f"🚨 Diyabet Riski Yüksek (%{bayes_pca_prediction_proba[0][1]*100:.1f} olasılık)")
      else:
          st.success(f"✅ Diyabet Riski Düşük (%{bayes_pca_prediction_proba[0][0]*100:.1f} olasılık)")
      st.progress(bayes_pca_prediction_proba[0][1])
      
      
      st.subheader("Logistic Regression Tahmin Sonucu")
      if lr_prediction[0] == 1:
          st.error(f"🚨 Diyabet Riski Yüksek (%{lr_prediction_proba[0][1]*100:.1f} olasılık)")
      else:
          st.success(f"✅ Diyabet Riski Düşük (%{lr_prediction_proba[0][0]*100:.1f} olasılık)")
      st.progress(lr_prediction_proba[0][1])


      st.subheader("Logistic Regression SMOTE Tahmin Sonucu")
      if best_prediction[0] == 1:
          st.error(f"🚨 Diyabet Riski Yüksek (%{best_prediction_proba[0][1]*100:.1f} olasılık)")
      else:
          st.success(f"✅ Diyabet Riski Düşük (%{best_prediction_proba[0][0]*100:.1f} olasılık)")
      st.progress(best_prediction_proba[0][1])