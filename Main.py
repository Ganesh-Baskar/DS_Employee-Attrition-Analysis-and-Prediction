import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load trained model
def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)

# Page config
st.set_page_config(page_title="🔍 Employee Attrition Prediction", layout="wide")

# Background and theme styling
st.markdown("""
    <style>
    .stApp {
        background-image: url("D:/Employment attrition prediction analysis/istockphoto-927747938-612x612.jpg");
        background-size: cover;
        background-attachment: fixed;
    }
    .stSidebar {
        background-color: rgba(173, 216, 230, 0.7); /* Light blue with transparency */
    }
    </style>
""", unsafe_allow_html=True)

# Load model
model = load_model('best_lgb_model.pkl')

# Sidebar navigation and file upload
with st.sidebar:
    st.title("📌 Navigation")
    with st.expander("🔎 Explore Pages"):
        page = st.radio("Go to", ["🏙️ Home", "📊 Exploratory Data Analysis", "🔮 Predict Attrition"])

    with st.expander("📁 Upload Data"):
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload a dataset to proceed.")
            st.stop()

# ---- Home Page ----
if page == "🏙️ Home":
    st.title("🚀 Employee Attrition Prediction App")
    st.markdown("""
### 🌟 Overview  
A powerful tool for HR teams to explore attrition trends and predict employee turnover using machine learning.

### 🎯 Goals
- 🧠 Analyze employee attrition
- 📈 Identify risk factors
- 🔮 Predict future attrition
- 🧰 Empower HR decisions

### 🛠️ Process
1. Data Collection  
2. Preprocessing  
3. ML Modeling (Best model: LightGBM)  
4. Prediction Interface  

### 📊 Insights
- Attrition by department & job role  
- Salary impact  
- Work-life balance influence  
- Satisfaction scores  
    """)

# ---- EDA Page ----
elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("👀 Dataset Preview")
    st.dataframe(df.head(5))

    st.subheader("📋 Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("📉 Attrition Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Attrition", data=df, palette="pastel", ax=ax1)
    st.pyplot(fig1)

    st.subheader("💰 Monthly Income vs. Attrition")
    fig2, ax2 = plt.subplots()
    sns.barplot(x="Attrition", y="MonthlyIncome", data=df, palette="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.subheader("🧑‍💼 Job Role vs. Attrition")
    fig3, ax3 = plt.subplots()
    sns.countplot(x="JobRole", hue="Attrition", data=df, palette="Set2", ax=ax3)
    plt.xticks(rotation=90)
    st.pyplot(fig3)

# ---- Prediction Page ----
elif page == "🔮 Predict Attrition":
    st.title("🔮 Predict Employee Attrition")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            Age = st.number_input("Age", 18, 65, value=30)
            DailyRate = st.number_input("Daily Rate", 100, 1500, value=500)
            DistanceFromHome = st.number_input("Distance From Home (km)", 1, 50, value=10)
            Education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
            TrainingTimesLastYear = st.number_input("Training Times Last Year", 0, 6, value=2)
            EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
            YearsAtCompany = st.number_input("Years at Company", 0, 40, value=5)
            HourlyRate = st.number_input("Hourly Rate", 10, 100, value=30)

        with col2:
            MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, value=5000)
            NumCompaniesWorked = st.number_input("Companies Worked", 0, 10, value=3)
            PercentSalaryHike = st.number_input("Salary Hike (%)", 5, 25, value=15)
            StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])
            WorkLifeBalance = st.selectbox("Work-Life Balance (1 = Poor, 4 = Excellent)", [1, 2, 3, 4])
            YearsInCurrentRole = st.number_input("Years in Current Role", 0, 20, value=4)
            YearsSinceLastPromotion = st.number_input("Years Since Promotion", 0, 20, value=2)

        # Categorical features
        BusinessTravel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Frequently", "Travel_Rarely"])
        Department = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])
        Gender = st.selectbox("Gender", ["Female", "Male"])
        OverTime = st.selectbox("OverTime", ["No", "Yes"])

        # Encoding
        cat_features = {
            "BusinessTravel_Non-Travel": int(BusinessTravel == "Non-Travel"),
            "BusinessTravel_Travel_Frequently": int(BusinessTravel == "Travel_Frequently"),
            "BusinessTravel_Travel_Rarely": int(BusinessTravel == "Travel_Rarely"),
            "Department_Human Resources": int(Department == "Human Resources"),
            "Department_Research & Development": int(Department == "Research & Development"),
            "Department_Sales": int(Department == "Sales"),
            "Gender_Female": int(Gender == "Female"),
            "Gender_Male": int(Gender == "Male"),
            "OverTime_No": int(OverTime == "No"),
            "OverTime_Yes": int(OverTime == "Yes"),
        }

        # Input assembly
        input_data = {
            **cat_features, "Age": Age, "DailyRate": DailyRate, "DistanceFromHome": DistanceFromHome,
            "Education": Education, "TrainingTimesLastYear": TrainingTimesLastYear,
            "EnvironmentSatisfaction": EnvironmentSatisfaction, "YearsAtCompany": YearsAtCompany,
            "HourlyRate": HourlyRate, "MonthlyIncome": MonthlyIncome, "NumCompaniesWorked": NumCompaniesWorked,
            "PercentSalaryHike": PercentSalaryHike, "StockOptionLevel": StockOptionLevel,
            "WorkLifeBalance": WorkLifeBalance, "YearsInCurrentRole": YearsInCurrentRole,
            "YearsSinceLastPromotion": YearsSinceLastPromotion
        }

        input_df = pd.DataFrame([input_data])

        # Align input to model
        expected_features = model.booster_.feature_name()
        for feature in expected_features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        input_df = input_df[expected_features]

        submit_btn = st.form_submit_button("📊 Predict")

        if submit_btn:
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[:, 1]
            if prediction[0] == 1:
                st.error(f"⚠️ High Attrition Risk! Probability: {probability[0]:.2f}")
            else:
                st.success(f"✅ Likely to Stay. Probability: {probability[0]:.2f}")