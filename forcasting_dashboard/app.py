import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Load Model + Data
rf = joblib.load("sales_model.pkl")
data = pd.read_csv("supermarket_dataset.csv")
data['Order_Date'] = pd.to_datetime(data['Order_Date'])

st.title("📊 Demand Forecasting Dashboard (Random Forest)")

categories = data['Product_Category'].unique()
selected_category = st.selectbox("Select Product Category", categories)

filtered_products = data[data['Product_Category'] == selected_category]['Product_Name'].unique()
selected_product = st.selectbox("Select Product", filtered_products)

min_date = data['Order_Date'].min().date()
max_date = data['Order_Date'].max().date()
date_range = st.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date)
)

# Filter data
filtered_data = data[
    (data['Product_Category'] == selected_category) &
    (data['Product_Name'] == selected_product) &
    (data['Order_Date'].dt.date >= date_range[0]) &
    (data['Order_Date'].dt.date <= date_range[1])
].copy()

st.subheader(f"📦 Showing Data for: {selected_product}")
st.write(filtered_data)

if filtered_data.empty:
    st.warning("No data available for this filter range.")
    st.stop()


# Feature Engineering
filtered_data['Year'] = filtered_data['Order_Date'].dt.year
filtered_data['Month'] = filtered_data['Order_Date'].dt.month
filtered_data['Day'] = filtered_data['Order_Date'].dt.day

categorical_cols = ['Region', 'Product_Category', 'Deal_Size']
numerical_cols = ['Year', 'Month', 'Day']

X_dashboard = pd.get_dummies(filtered_data[categorical_cols + numerical_cols], drop_first=True)

# Match model feature order
for col in rf.feature_names_in_:
    if col not in X_dashboard.columns:
        X_dashboard[col] = 0
X_dashboard = X_dashboard[rf.feature_names_in_]


# Forecasting
filtered_data['Forecast'] = rf.predict(X_dashboard)

# Add moving average for trend
filtered_data['MA_7'] = filtered_data['Units_Sold'].rolling(window=7, min_periods=1).mean()
filtered_data['Forecast_MA_7'] = filtered_data['Forecast'].rolling(window=7, min_periods=1).mean()


# KPIs
total_actual = int(filtered_data['Units_Sold'].sum())
total_forecast = int(filtered_data['Forecast'].sum())
perc_diff = ((total_forecast - total_actual) / total_actual) * 100 if total_actual != 0 else 0

st.subheader("📌 Key Performance Indicators")
col1, col2, col3 = st.columns(3)
col1.metric("Total Actual Sales", total_actual)
col2.metric("Total Forecasted Sales", total_forecast)
col3.metric("% Difference", f"{perc_diff:.2f}%")


# Feature Importance
st.subheader("🔥 Top Contributing Features")
feature_importances = pd.DataFrame({
    "Feature": X_dashboard.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)
st.bar_chart(feature_importances.set_index("Feature"))


# Chart: Actual vs Forecast
st.subheader("📈 Actual vs Forecast Trend")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=filtered_data['Order_Date'],
    y=filtered_data['Units_Sold'],
    mode='lines+markers',
    name='Actual Sales',
    line=dict(color='blue', width=3)
))
fig.add_trace(go.Scatter(
    x=filtered_data['Order_Date'],
    y=filtered_data['Forecast'],
    mode='lines+markers',
    name='Forecast',
    line=dict(color='orange', width=3, dash='dash')
))
# Moving averages
fig.add_trace(go.Scatter(
    x=filtered_data['Order_Date'],
    y=filtered_data['MA_7'],
    mode='lines',
    name='7-Day Actual MA',
    line=dict(color='green', width=2)
))
fig.add_trace(go.Scatter(
    x=filtered_data['Order_Date'],
    y=filtered_data['Forecast_MA_7'],
    mode='lines',
    name='7-Day Forecast MA',
    line=dict(color='red', width=2, dash='dot')
))

fig.update_layout(
    title=f"Actual vs Forecast for {selected_product}",
    xaxis_title="Date",
    yaxis_title="Units Sold",
    template="plotly_white",
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig, use_container_width=True)


st.subheader("📊 Sales Distribution by Region")
region_sales = filtered_data.groupby('Region')['Units_Sold'].sum().reset_index()
fig2 = px.pie(region_sales, names='Region', values='Units_Sold',
              title='Sales by Region', color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig2, use_container_width=True)


# Download CSV
st.download_button(
    label="⬇ Download Forecast CSV",
    data=filtered_data.to_csv(index=False),
    file_name=f"{selected_product}_forecast.csv",
    mime="text/csv"
)
