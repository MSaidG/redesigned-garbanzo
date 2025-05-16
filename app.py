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
)
st.write(df['Outcome'].value_counts())

# Streamlit UI
st.title("ML Model Deployment")
st.write("Enter features below:")

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

# Load the model
bayes_pca_model = load_bayes_pca_model()
lr_model = load_lr_model()
lda = joblib.load("lda.pkl")
pca = joblib.load("pca.pkl")
scaler = joblib.load("scaler.pkl")

features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]

if st.button("Predict"):

    input_array = np.array(features).reshape(1, -1) # Shape: 1, 8
    input_scaled = scaler.transform(input_array) 
    input_pca = pca.transform(input_scaled)
    
    bayes_pca_prediction = bayes_pca_model.predict(input_pca)
    st.success(f"Bayes PCA Tahmin: {bayes_pca_prediction[0]}")
    bayes_pca_prediction_proba = bayes_pca_model.predict_proba(input_pca)
    
    lr_prediction = lr_model.predict(input_scaled)
    st.success(f"Logistic Regression Tahmin: {lr_prediction[0]}")
    lr_prediction_proba = lr_model.predict_proba(input_scaled)
    
    
    
    # # Girdileri düzenle
    # input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    
    # st.write(input_data)
    # # Tahmin yap
    # prediction = model.predict(input_scaled)
    # prediction_proba = model.predict_proba(input_scaled)
    
    # # Sonuçları göster
    # if prediction[0] == 1:
    #     st.error(f"Tahmin: DİYABET (Olasılık: %{prediction_proba[0][1]*100:.2f})")
    # else:
    #     st.success(f"Tahmin: DİYABET DEĞİL (Olasılık: %{prediction_proba[0][0]*100:.2f})")
        
    #     # Modelin rastgele bir örnekteki davranışını test edin
    # test_sample = np.array([[1, 90, 70, 25, 80, 22, 0.3, 25]])  # Açıkça negatif bir örnek
    # print(model.predict(test_sample))  # 0 vermeli!





    # # Ölçeklendirme
    # input_array = np.array(features).reshape(1, -1) # Shape: 1, 8
    # input_scaled = scaler.transform(input_array)
    
    # # Tahmin
    # prediction = model.predict(input_scaled)
    # prediction_proba = model.predict_proba(input_scaled)
    
    # Sonuçları gösterme
    st.subheader("Bayes PCA Tahmin Sonucu")
    if bayes_pca_prediction[0] == 1:
        st.error(f"🚨 Diyabet Riski Yüksek (%{bayes_pca_prediction_proba[0][1]*100:.1f} olasılık)")
    else:
        st.success(f"✅ Diyabet Riski Düşük (%{bayes_pca_prediction_proba[0][0]*100:.1f} olasılık)")
    
    # Olasılık çubuğu
    st.progress(bayes_pca_prediction_proba[0][1])
    
    st.subheader("Logistic Regression Tahmin Sonucu")
    if lr_prediction[0] == 1:
        st.error(f"🚨 Diyabet Riski Yüksek (%{lr_prediction_proba[0][1]*100:.1f} olasılık)")
    else:
        st.success(f"✅ Diyabet Riski Düşük (%{lr_prediction_proba[0][0]*100:.1f} olasılık)")
    
    # Olasılık çubuğu
    st.progress(lr_prediction_proba[0][1])
