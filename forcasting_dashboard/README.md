
# 📊Forecasting Dashboard (Random Forest)

A simple and interactive **Streamlit-based dashboard** for forecasting product demand using a **Random Forest Regression model**.
The app allows users to explore sales data, apply filters, visualize trends, and download forecasted results.

---

## 🚀 Features

* **Product Category & Product selection**
* **Custom Date Range filtering**
* **Actual vs Forecast Trend visualization**
* **7-Day Moving Average trends**
* **Top Feature Importances**
* **Regional Sales Distribution (Pie Chart)**
* **Forecast CSV download**
* Clean UI powered by **Streamlit + Plotly**

---

## 🧠 Model Used

* **Random Forest Regressor**
* Trained using supermarket sales dataset
* Handles categorical features via **one-hot encoding**
* Feature importance shown inside the dashboard

---

## 📂 Project Structure

```
├── supermarket_dataset.csv
├── sales_model.pkl
├── app.py               # Streamlit dashboard
└── README.md
```

---

## ▶️ Run the Dashboard

Make sure you have the required dependencies:

```bash
pip install streamlit pandas plotly joblib
```

Run the app:

```bash
streamlit run app.py
```

---

## 📈 Output Screens (Main Highlights)

* Actual vs Forecast line chart
* Moving averages
* KPIs (Actual, Forecast, % Difference)
* Feature importance bar chart
* Sales by region pie chart

---

## 📥 Download

Users can download the **forecasted results as CSV** directly from the dashboard.
