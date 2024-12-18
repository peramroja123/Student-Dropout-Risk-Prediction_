import streamlit as st
import pandas as pd
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# App Header
st.header("📚 Student Dropout Risk Prediction 📉")
st.write("📊 *Predictive Model Built on Student Data*")

# Load the dataset and clean it
try:
    df = pd.read_csv("simulated_dropout_prediction_dataset.csv")

    # Explicitly remove "Student ID" column if it exists
    if 'Student_ID' in df.columns:
        df = df.drop(columns=['Student_ID'])
    
    # Remove "Unnamed" columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Rename the column from Test_Scores to Test_Scores_median
    if 'Test_Scores' in df.columns:
        df = df.rename(columns={'Test_Scores': 'Test_Scores_median'})

    st.dataframe(df.head())  # Display the first few rows of the cleaned dataset
except FileNotFoundError:
    st.error("❌ Dataset file 'simulated_dropout_prediction_dataset.csv' not found. Please upload the file.")
    st.stop()

# Input fields for user data
st.write("🌟 *Enter the following student data for prediction:*")

# Layout input fields using columns
col1, col2 = st.columns(2)

with col1:
    # Gender: Categorical input, select from labels
    Gender = st.selectbox("Select Gender", options=["Female", "Male"])
    # Map selected gender to numeric values
    Gender = 0 if Gender == "Female" else 1

with col2:
    # Parental Education: Categorical input, select from labels
    Parental_Education = st.selectbox("Select Parental Education Level", options=["Uneducated", "Primary", "Secondary", "Tertiary"])
    # Map selected education to numeric values
    Parental_Education = {"Uneducated": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3}[Parental_Education]

col3, col4 = st.columns(2)
with col3:
    # Attendance Rate: Numerical input, enter percentage
    Attendance_Rate = st.number_input("Enter Attendance Rate (%)", min_value=0.0, max_value=100.0, step=0.1)

with col4:
    # SocioEconomic Status: Categorical input, select from labels
    SocioEconomic_Status = st.selectbox("Select Socioeconomic Status", options=["Low", "Medium", "High"])
    # Map selected status to numeric values
    SocioEconomic_Status = {"Low": 0, "Medium": 1, "High": 2}[SocioEconomic_Status]

col5, col6 = st.columns(2)
with col5:
    # Engagement Score: Numerical input, enter value between 0 and 100
    Engagement_Score = st.number_input("Enter Engagement Score (0 to 100)", min_value=0.0, max_value=100.0, step=0.1)

with col6:
    # Test Scores Median: Numerical input, enter value between 0 and 100
    Test_Scores_median = st.number_input("Enter Test Scores Median (0 to 100)", min_value=0.0, max_value=100.0, step=0.1)

# Prepare input data for prediction
input_data = [
    [Gender, Parental_Education, Attendance_Rate, SocioEconomic_Status, Engagement_Score, Test_Scores_median]
]

###################### Prediction Logic ######################

# Load the trained model
try:
    model = joblib.load('Dropout_pred.pkl')
except FileNotFoundError:
    st.error("❌ Model file 'Dropout_pred.pkl' not found. Please upload the model.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Predict and display result when button is clicked
if st.button("🔍 Predict Dropout Risk"):
    try:
        # Ensure all input data is valid (categorical data as integers and numerical data as floats)
        input_data = [[int(Gender), int(Parental_Education), float(Attendance_Rate), int(SocioEconomic_Status), float(Engagement_Score), float(Test_Scores_median)]]
        
        prediction = model.predict(input_data)
        
        # Display result with a message
        if prediction[0] == 1:
            st.markdown("<h3 style='color: red;'>The student is at risk of dropping out. 🚨</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: green;'>The student is not at risk of dropping out. ✅</h3>", unsafe_allow_html=True)
        
        # Optionally, show the input data
        st.write("💾 *Given Input:*")
        input_df = pd.DataFrame(input_data, columns=[
            "Gender", "Parental Education", "Attendance Rate", 
            "Socioeconomic Status", "Engagement Score", "Test Scores Median"
        ])
        st.dataframe(input_df)
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
