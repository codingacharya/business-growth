# streamlit_full_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

st.set_page_config(page_title="Autonomous Corporate Strategy Engine", layout="wide")
st.title("🚀 Autonomous Corporate Strategy Engine")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train_columns' not in st.session_state:
    st.session_state.X_train_columns = None
if 'target' not in st.session_state:
    st.session_state.target = None

# Upload CSV
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(20))
    
    st.write("Columns detected:", list(df.columns))

    # Sidebar: Visualization options
    st.sidebar.header("Visualization Options")
    chart_type = st.sidebar.selectbox("Choose chart type", ["Line Chart", "Bar Chart", "Heatmap", "Competitor Comparison"])
    metric = st.sidebar.selectbox("Select Metric", df.columns[2:5])

    st.subheader("📊 Visualizations")
    if chart_type == "Line Chart":
        plt.figure(figsize=(12,6))
        sns.lineplot(data=df, x="Date", y=metric, hue="Company", marker="o")
        plt.xticks(rotation=45)
        st.pyplot(plt)
    elif chart_type == "Bar Chart":
        plt.figure(figsize=(12,6))
        sns.barplot(data=df, x="Company", y=metric, hue="Date")
        st.pyplot(plt)
    elif chart_type == "Heatmap":
        plt.figure(figsize=(10,6))
        corr = df.select_dtypes(include=np.number).corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        st.pyplot(plt)
    elif chart_type == "Competitor Comparison":
        plt.figure(figsize=(12,6))
        sns.barplot(data=df, x="Date", y=metric, hue="Company")
        st.pyplot(plt)

    # Predictive Modeling
    st.subheader("🤖 Predictive Modeling")
    default_features = ["Stock Price","Marketing Spend (M USD)","Consumer Sentiment (1-10)"]
    default_features = [f for f in default_features if f in df.columns[2:-1]]

    features = st.multiselect("Select Features for Prediction", df.columns[2:-1], default=default_features)
    target = st.selectbox("Select Target Variable", ["Revenue (M USD)", "Market Share (%)"])
    
    if st.button("Run Prediction"):
        df_model = pd.get_dummies(df, columns=["Company"], drop_first=True)
        X = df_model[[f for f in features if f in df_model.columns] + [col for col in df_model.columns if "Company_" in col]]
        y = df_model[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Store model and info in session state
        st.session_state.model = model
        st.session_state.X_train_columns = X_train.columns
        st.session_state.target = target

        # Metrics
        st.write(f"**Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"**Mean Absolute Error (MAE):** {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

        # Plot predictions
        plt.figure(figsize=(10,6))
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{target} Prediction")
        st.pyplot(plt)

    # Scenario Simulation
    if st.session_state.model:
        st.subheader("⚡ Scenario Simulation")
        company_options = df['Company'].unique()
        selected_company = st.selectbox("Select Company for Simulation", company_options)
        
        stock_col = "Stock Price"
        marketing_col = "Marketing Spend (M USD)"
        sentiment_col = [col for col in df.columns if "Consumer Sentiment" in col][0]

        marketing_spend = st.slider("Marketing Spend (M USD)", float(df[marketing_col].min()), float(df[marketing_col].max()), float(df[marketing_col].mean()))
        sentiment = st.slider("Consumer Sentiment", int(df[sentiment_col].min()), int(df[sentiment_col].max()), int(df[sentiment_col].mean()))

        if st.button("Simulate Scenario"):
            # Build sim_df with same columns as model
            sim_df = pd.DataFrame(columns=st.session_state.X_train_columns)
            sim_df.loc[0] = 0  # initialize with zeros

            # Fill numeric features
            if stock_col in sim_df.columns:
                sim_df.at[0, stock_col] = df[df["Company"]==selected_company][stock_col].mean()
            if marketing_col in sim_df.columns:
                sim_df.at[0, marketing_col] = marketing_spend
            if sentiment_col in sim_df.columns:
                sim_df.at[0, sentiment_col] = sentiment

            # Fill company dummy columns
            for col in sim_df.columns:
                if col.startswith("Company_"):
                    if col == f"Company_{selected_company}":
                        sim_df.at[0, col] = 1

            # Predict
            sim_pred = st.session_state.model.predict(sim_df)
            st.success(f"Predicted **{st.session_state.target}** for {selected_company}: **{sim_pred[0]:.2f}**")
    else:
        st.info("Run prediction first to enable scenario simulation.")
else:
    st.info("Please upload a CSV file to start analysis.")
