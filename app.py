import streamlit as st
import pandas as pd
from dataset_1_xgboost import get_ds_1_model, process_data_model_1
from dataset_2_logistic import get_ds_2_model,process_data_model_2
ds_1_model=get_ds_1_model()
ds_2_model=get_ds_2_model()

# Assuming you have your custom machine learning models as functions
def predict_heart_disease_model1(data):
    processed_data=process_data_model_1(data)
    prediction=ds_1_model.predict(processed_data)
    return prediction

def predict_heart_disease_model2(data):
    processed_data=process_data_model_2(data)
    prediction=ds_2_model.predict(processed_data)
    return prediction

# Function to create the UI for Tab 1
def tab1_ui():
    st.title("Framingham_dataset_1")
    sbp = st.number_input("Systolic Blood Pressure (sbp)", min_value=0)
    tobacco = st.number_input("Tobacco", min_value=0.0)
    ldl = st.number_input("LDL Cholesterol", min_value=0.0)
    adiposity = st.number_input("Adiposity", min_value=0.0)
    famhist = st.selectbox("Family History", ["Present", "Absent"])
    typea = st.number_input("Type A Behavior", min_value=0)
    obesity = st.number_input("Obesity", min_value=0.0)
    alcohol = st.number_input("Alcohol", min_value=0.0)
    age = st.number_input("Age", min_value=0)

    if st.button("Submit"):
        # Collect the user input data
        user_data = {
            "sbp": sbp,
            "tobacco": tobacco,
            "ldl": ldl,
            "adiposity": adiposity,
            "famhist": famhist,
            "typea": typea,
            "obesity": obesity,
            "alcohol": alcohol,
            "age": age
        }

        # Call the custom machine learning model function
        prediction = predict_heart_disease_model1(user_data)

        # Display the prediction
        st.success(f"Prediction: {prediction}")

# Function to create the UI for Tab 2
def tab2_ui():
    st.title("Framingham_dataset_2")
    gender = st.selectbox("Male", [0, 1])
    age = st.number_input("Age", min_value=0)
    education = st.number_input("Education", min_value=0)
    current_smoker = st.selectbox("Current Smoker", [0,1])
    cigs_per_day = st.number_input("Cigarettes per Day", min_value=0)
    bp_meds = st.selectbox("Blood Pressure Medication", [0,1])
    prevalent_stroke = st.selectbox("Prevalent Stroke", [0,1])
    prevalent_hyp = st.selectbox("Prevalent Hypertension", [0,1])
    diabetes = st.selectbox("Diabetes", [0,1])
    tot_chol = st.number_input("Total Cholesterol", min_value=0.0)
    sys_bp = st.number_input("Systolic Blood Pressure", min_value=0)
    dia_bp = st.number_input("Diastolic Blood Pressure", min_value=0)
    BMI = st.number_input("Body Mass Index (BMI)", min_value=0.0)
    heart_rate = st.number_input("Heart Rate", min_value=0)
    glucose = st.number_input("Glucose", min_value=0.0)

    if st.button("Submit"):
        # Collect the user input data
        user_data = {
            "male": gender,
            "age": age,
            "education": education,
            "currentSmoker": current_smoker,
            "cigsPerDay": cigs_per_day,
            "BPMeds": bp_meds,
            "prevalentStroke": prevalent_stroke,
            "prevalentHyp": prevalent_hyp,
            "diabetes": diabetes,
            "totChol": tot_chol,
            "sysBP": sys_bp,
            "diaBP": dia_bp,
            "BMI": BMI,
            "heartRate": heart_rate,
            "glucose": glucose
        }

        # Call the custom machine learning model function
        prediction = predict_heart_disease_model2(user_data)

        # Display the prediction
        st.success(f"Prediction: {prediction}")

# Main Streamlit app
def main():
    st.set_page_config(page_title="Heart Disease Prediction", page_icon=":heart:")
    st.title("Heart Disease Prediction")

    # Create tabs in the navbar
    tabs = ["Framingham_dataset_1", "Framingham_dataset_2"]
    selected_tab = st.sidebar.radio("Select Dataset", tabs)

    # Display the selected tab's UI
    if selected_tab == "Framingham_dataset_1":
        tab1_ui()
    elif selected_tab == "Framingham_dataset_2":
        tab2_ui()

if __name__ == "__main__":
    main()
