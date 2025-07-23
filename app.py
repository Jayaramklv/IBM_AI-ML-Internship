import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.title("💼 Employee Salary Range Prediction")

# ----- Custom label maps -----
gender_map = {"Female": 0, "Male": 1}
education_map = {"Bachelors": 0, "Masters": 1, "PhD": 2, "High School": 3}  # Adjust if you used 4 levels
job_map = {
    "Business Development Manager": 0, "Customer Service Representative": 1, "Digital Marketing Manager": 2, "Director of Marketing": 3,
    "Director of Operations": 4, "IT Manager": 5, "Junior Business Analyst": 6, "Junior Business Development Associate": 7,
    "Junior Financial Analyst": 8, "Junior Marketing Coordinator": 9, "Junior Marketing Specialist": 10, "Junior Operations Analyst": 11,
    "Junior Product Manager": 12, "Junior Project Manager": 13, "Junior Sales Representative": 14, "Junior Web Developer": 15,
    "Senior Business Analyst": 16, "Senior Business Development Manager": 17, "Senior Data Engineer": 18, "Senior Data Scientist": 19,
    "Senior Financial Analyst": 20, "Senior Financial Manager": 21, "Senior Marketing Analyst": 22, "Senior Marketing Specialist": 23,
    "Senior Operations Coordinator": 24, "Senior Operations Manager": 25, "Senior Product Designer": 26, "Senior Product Manager": 27,
    "Senior Project Coordinator": 28, "Senior Project Manager": 29, "Senior Software Engineer": 30
}
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📁 Bulk Prediction"])

# ----- Streamlit UI -----
with tab1:
    # ----- Salary Range Info -----
    st.markdown("### 💰 Salary Range")
    st.table({
        "Range Label": ['<50k', '<1L', '<1.5L', '<2L'],
        "Salary Range (INR)": ['0 - 50,000', '50,001 - 100,000', '100,001 - 150,000', '150,001 - 200,000']
    })
    st.subheader("Enter Employee Details")
    Age = st.slider("Enter Age", min_value=18, max_value=80, value=30)
    Gender_display = st.selectbox("Select Gender", options=list(gender_map.keys()))
    Education_display = st.selectbox("Select Education Level", options=list(education_map.keys()))
    Job_display = st.selectbox("Select Job Title", options=list(job_map.keys()))
    HoursWorkedPerWeek = st.slider("Hours Worked Per Week", min_value=0, max_value=120, value=60)
    Years_of_Experience = st.slider("Years of Experience", min_value=0, max_value=50, value=5)

    # ----- Prediction -----
    if st.button("Predict Salary Range"):
        try:
            # Convert to encoded values
            Gender = gender_map[Gender_display]
            Education_Level = education_map[Education_display]
            Job_Title = job_map[Job_display]

            # Prepare input
            input_data = pd.DataFrame([[
                Age, Gender, Education_Level, Job_Title,
                HoursWorkedPerWeek, Years_of_Experience
            ]], columns=['age', 'Gender', 'Education_Level', 'Job_Title', 
                        'HoursWorkedPerWeek', 'Years_of_Experience'])

            # Predict
            prediction = model.predict(input_data)
            st.success(f"✅ Predicted Salary Range: **{prediction[0]}**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
with tab2:
            # Batch prediction
    st.markdown("#### 📂 Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:", batch_data.head())
        batch_preds = model.predict(batch_data)
        batch_data['PredictedClass'] = batch_preds
        st.write("✅ Predictions:")
        st.write(batch_data.head())
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
