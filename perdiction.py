import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Ad Block Click Prediction", page_icon="📈")

st.title("Ad Block Click Prediction 📈")
                    
# Load dataset
def load_data():
    df = pd.read_csv("advertising.csv")  # Ensure this dataset has appropriate columns
    return df

df = load_data()

# Display dataset
st.header("Ad Block Click Dataset")
st.write(df)

# Select features and target
X = df.select_dtypes(include=['number']).drop(columns=['Clicked on Ad'])  # Use only numeric features
y = df['Clicked on Ad']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# User Input Features
st.header("User Input Features")
def user_input_features():
    input_data = {}
    for col in X.columns:
        input_data[col] = st.slider(f'{col}', float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    return pd.DataFrame(input_data, index=[0])

input_df = user_input_features()

# Model selection
model_type = st.selectbox("Select Model", ["Naïve Bayes", "Decision Tree", "Linear Regression", "Random Forest"])

# Model initialization
if model_type == "Naïve Bayes":
    model = GaussianNB()
elif model_type == "Decision Tree":
    model = DecisionTreeRegressor()
elif model_type == "Linear Regression":
    model = LinearRegression()
elif model_type == "Random Forest":
    model = RandomForestRegressor()

# Fit model
model.fit(X_train, y_train)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.subheader('Prediction')
    st.write(f"Predicted Ad Click Probability: **{prediction[0]:.2f}**")
 
