from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import streamlit as st
import numpy as np
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
import pandas as pd
import joblib


# Veri yükleme ve ön işleme (önbelleğe alınmış)
@st.cache_data
def load_and_preprocess_data():
    diabetes_data_copy = pd.read_csv("diabetes.csv")  # Verinizi yükleyin
    X = diabetes_data_copy.drop(["Outcome"], axis=1)
    y = diabetes_data_copy["Outcome"]
    return X, y

X, y = load_and_preprocess_data()

# Model eğitimi (önbelleğe alınmış)
@st.cache_resource
def train_model():
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    mlp = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=5000, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    
    return mlp, scaler, X_test_scaled, y_test

model, scaler, X_test, y_test = train_model()

# Load CSV data
@st.cache_data  # Cache to avoid reloading on every interaction
def load_data():
    return pd.read_csv("diabetes.csv")  # Replace with your file path
  
# Modeli yükle
@st.cache_resource
def load_model():
    return joblib.load("lr_lda_model.pkl")


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

# Streamlit UI
st.title("ML Model Deployment")
st.write("Enter features below:")

# Check selected rows
selected_rows = grid_response["selected_rows"]
if selected_rows is not None and not selected_rows.empty:
    pregnancies = st.number_input("Pregnancies", value=selected_rows.iloc[0]["Pregnancies"])  
    glucose = st.number_input("Glucose", value=selected_rows.iloc[0]["Glucose"])  
    blood_pressure = st.number_input("Blood Pressure", value=selected_rows.iloc[0]["BloodPressure"])  
    skin_thickness = st.number_input("Skin Thickness(mm)", value=selected_rows.iloc[0]["SkinThickness"])  
    insulin = st.number_input("Insulin(mu U/ml)", value=selected_rows.iloc[0]["Insulin"]) 
    bmi = st.number_input("BMI", value=selected_rows.iloc[0]["BMI"])  
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", value=selected_rows.iloc[0]["DiabetesPedigreeFunction"])  
    age = st.number_input("Age", value=selected_rows.iloc[0]["Age"])  
else:
    pregnancies = st.number_input("Pregnancies", value=0)
    glucose = st.number_input("Glucose", value=0)
    blood_pressure = st.number_input("Blood Pressure", value=0)
    skin_thickness = st.number_input("Skin Thickness(mm)", value=0)
    insulin = st.number_input("Insulin(mu U/ml)", value=0)
    bmi = st.number_input("BMI", value=0)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", value=0)
    age = st.number_input("Age", value=0)

# Load the model
model = load_model()
lda = joblib.load("lda.pkl")
scaler = joblib.load("scaler.pkl")

features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]

# st.write("Veri Dağılımı:", df["Outcome"].value_counts())
print(df['Outcome'].value_counts())
if st.button("Predict"):

    input_array = np.array(features).reshape(1, -1) # Shape: 1, 8
    input_scaled = scaler.transform(input_array) 
    input_lda = lda.transform(input_scaled)
    
    prediction = model.predict(input_lda)
    st.success(f"Prediction: {prediction[0]}")
    prediction_proba = model.predict_proba(input_lda)
    
    
    
    
    
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
    st.subheader("Tahmin Sonucu")
    if prediction[0] == 1:
        st.error(f"🚨 Diyabet Riski Yüksek (%{prediction_proba[0][1]*100:.1f} olasılık)")
    else:
        st.success(f"✅ Diyabet Riski Düşük (%{prediction_proba[0][0]*100:.1f} olasılık)")
    
    # Olasılık çubuğu
    st.progress(prediction_proba[0][1])
