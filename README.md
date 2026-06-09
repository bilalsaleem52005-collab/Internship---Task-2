Task 1-
# 🌸 1- Term Deposit Prediction – Training Pipeline

**3 models + SHAP explanations + automatic plots**

Predict if a bank client will subscribe to a term deposit.

---

## 📌 What & Why

- **Business need**: Banks run campaigns – only ~11.7% of clients subscribe.  
  Predicting “yes” helps target the right customers.

- **My solution**: Train & compare 3 models, explain predictions with SHAP.

---

## 🧠 What I Did (Step by Step)

| Step | Action | Why |
|------|--------|-----|
| 1 | Load & explore data | Check shape, missing values, class balance |
| 2 | Preprocess | One‑hot encode categories, standard scale numbers |
| 3 | Train 3 models | Logistic Regression (baseline), Random Forest, XGBoost |
| 4 | Evaluate | Accuracy, F1‑score, AUC‑ROC, confusion matrices, ROC curves |
| 5 | Compare | Table shows XGBoost performs best |
| 6 | Feature importance | Top 15 features for RF & XGB |
| 7 | SHAP explanations | Global summary + 5 waterfall plots (subsampled for speed) |
| 8 | Save best model | `best_model.pkl` for later use |

---

## 📊 Results (on test set)

| Model                 | Accuracy | F1‑Score | AUC‑ROC |
|-----------------------|----------|----------|---------|
| Logistic Regression   | 0.891    | 0.507    | 0.908   |
| Random Forest         | 0.902    | 0.580    | 0.931   |
| **XGBoost**           | **0.905**| **0.585**| **0.936** |

> **Top features** (from SHAP): `duration`, `poutcome_success`, `month_may`, `housing_yes`, `age`

---
---

pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------
----------------------------------------------
# 🛍️ 2- Customer Segmentation – Mall Customers

**Unsupervised Learning: K‑Means + PCA/t‑SNE visualisation + actionable marketing strategies**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

A mall wants to understand its customers better – not just who they are, but **how they behave** (spending vs income).  
Using **unsupervised learning**, we segment customers into meaningful groups and propose targeted marketing strategies.

**What this project does:**
- Exploratory Data Analysis (age, income, spending, gender)
- K‑Means clustering to find natural customer segments
- PCA & t‑SNE visualisations of the clusters
- Interpret each segment and recommend marketing actions

---

## 🧠 What I Did (Step by Step)

| Step | Action | Why |
|------|--------|-----|
| 1 | Load & explore data | Check shape, missing values, distributions |
| 2 | Select features | `Annual Income` + `Spending Score` – the two most behaviour‑rich columns |
| 3 | Scale features | StandardScaler – income (k$) and spending (0‑100) live on different scales |
| 4 | Find optimal k | Elbow method + silhouette score → k = 5 |
| 5 | Run K‑Means | Segment customers into 5 clusters |
| 6 | Visualise | Scatter plots (original space, PCA, t‑SNE) |
| 7 | Profile clusters | Average age, income, spending, dominant gender per cluster |
| 8 | Suggest strategies | Tailored marketing actions for each segment |
| 9 | Save outputs | All plots + CSV summary |

---

## 📊 Results & Insights

### Optimal Number of Clusters: k = 5

| Cluster | Age | Income (k$) | Spending Score | Dominant Gender | Segment | Marketing Strategy |
|---------|-----|-------------|----------------|-----------------|---------|---------------------|
| 0 | 42.7 | 55.3 | 49.5 | Female | Standard (mid income, mid spending) | Loyalty programs, cross‑sell |
| 1 | 41.1 | 88.2 | 17.1 | Male | High income, low spending (price sensitive) | Show value, free trials |
| 2 | 45.2 | 26.3 | 20.9 | Female | Low income, low spending (budget) | Discounts, bundle deals |
| 3 | 25.3 | 25.7 | 79.4 | Female | Low income, high spending (aspirational) | BNPL, aspirational ads |
| 4 | 32.7 | 86.5 | 82.1 | Female | High income, high spending (VIP) | Premium products, exclusive events |

### Visualisations (saved automatically)

- **EDA distributions** – age, income, spending, gender  
- **Elbow & silhouette** – proving k=5 is optimal  
- **Clusters (original space)** – income vs spending coloured by cluster  
- **PCA projection** – linear dimensionality reduction  
- **t‑SNE projection** – better separation of clusters  
- **Centroids** – red X marks the cluster centers  

---

## Task 5- Global Superstore – Interactive Dashboard

### 1. Install dependencies
```bash pip install pandas numpy matplotlib seaborn scikit-learn```

-------------------------------
# Global Superstore – Premium Interactive Dashboard

## 🌟 What This Project Does

This is a **fully interactive sales analytics dashboard** built with Python. It lets you explore the famous Global Superstore dataset – filter by date, region, product category, and customer segment – and see key metrics (sales, profit, margin, discount, units, orders) update in real time. All charts are interactive: you can hover, zoom, pan, and toggle between **Dark and Light themes** with one click.

**In plain English:** It turns a boring CSV file into a beautiful, business‑intelligence tool that anyone can use without writing code.

---

## 🧠 How I Made This Project – Step by Step

### Step 0 – The Idea
I wanted to demonstrate modern data dashboarding skills. The goal was to build something that looks like a SaaS product, not a script. I decided on the Superstore dataset because it’s rich enough for real analytics but small enough to run instantly.

### Step 1 – Choosing the Right Tools
I evaluated several frameworks:

| Tool | Why I chose it (or not) |
|------|------------------------|
| **Streamlit** | Winner! Turns Python into a web app with zero front‑end code. Has built‑in widgets, caching, and layout. |
| Plotly | Best for interactive charts. Hover tooltips, zoom, and theme control. |
| Pandas | Essential for filtering, grouping, and aggregations. |
| CSS (custom) | Streamlit’s default style is basic. I wanted glassmorphism and a theme toggle, so I wrote custom CSS. |

Why not Tableau/Power BI? They are expensive, not code‑friendly, and hard to version control.  
Why not Dash? Too much boilerplate (callbacks, HTML layout).  
Why not R Shiny? I’m a Python user.

### Step 2 – Setting Up the Data Pipeline
- Used `@st.cache_data` to download the CSV only once and cache it.
- Cleaned column names, parsed dates, dropped invalid rows.
- This makes the app fast even on repeated runs.

### Step 3 – Designing the UI/UX
I sketched a layout with:
- Sidebar (filters + theme toggle)
- Top row: KPI cards (6 metrics)
- Then rows of charts: time series, top customers, category pie, sub‑category treemap, region bars, segment bars.
- Footer with download and data table.

I deliberately used **glassmorphism** (semi‑transparent cards, blur, soft shadows) because it looks modern and premium.

### Step 4 – Building the Theme Toggle
This was the hardest part. Streamlit reruns the whole script on every interaction, so I used `st.session_state` to remember the theme. I wrote **two full CSS blocks** (dark and light) and inject the right one. I also wrote a helper function to update Plotly figure colors (background, grid, fonts) based on the current theme.

### Step 5 – Implementing Filters
I added:
- Date range picker
- Region (multiselect)
- Category (multiselect)
- Sub‑category (dynamic – updates based on selected category)
- Segment (if column exists)

All filters modify a single `filtered_df` dataframe. Every chart then recomputes from that dataframe.

### Step 6 – Creating the Charts
For each chart:
- Aggregate data with Pandas (groupby, sum, etc.)
- Create Plotly figure
- Apply theme styling
- Display with `st.plotly_chart()`

The time series uses `pd.Grouper(freq='M')` to group by month. The treemap uses `px.treemap`. The KPI cards are custom HTML/CSS inside `st.markdown(..., unsafe_allow_html=True)`.

### Step 7 – Polish & Error Handling
- Added `st.info()` for empty states (e.g., no data for selected filters).
- Made sure missing columns (like `Segment` in some datasets) don’t crash the app.
- Used `use_container_width=True` so all charts resize nicely.

### Step 8 – Packaging
Saved as a single Python file. Added a `requirements.txt` and this README.

---

## 🛠️ Technologies Used & Why

| Technology | Purpose | Why this choice |
|------------|---------|----------------|
| **Streamlit** | Web framework | Fastest way to go from Python script to interactive dashboard. No HTML/JS needed. |
| **Pandas** | Data manipulation | Essential for filtering, grouping, aggregations. Handles large data efficiently. |
| **Plotly** | Interactive charts | Hover info, zoom, pan, and beautiful defaults. Easy theming. |
| **Matplotlib/Seaborn** | Fallback | Installed but not used; kept for compatibility. |
| **NumPy** | Numerical ops | Margin calculations, percentage formatting. |
| **CSS (custom)** | Styling | Streamlit’s default is plain. CSS gives glassmorphism, rounded corners, gradients, and dual theme. |
| **Session State** | State management | To remember dark/light theme across reruns. |

---

## 🧰 Skills Demonstrated

This project showcases the following skills:

| Skill Area | Specifics |
|------------|-----------|
| **Python** | Writing clean, modular code. Using decorators (`@st.cache_data`). Working with dates, strings, and numbers. |
| **Data Analysis (Pandas)** | Filtering, grouping, aggregation, handling missing data, time series resampling. |
| **Data Visualization (Plotly)** | Line charts, bar charts, pie charts, treemaps. Customizing colors, gridlines, fonts, and themes. |
| **Web App Development** | Building an interactive UI with Streamlit: sidebar, columns, expanders, buttons, download widgets. |
| **Front‑end Styling** | Writing custom CSS to override default Streamlit styles. Implementing glassmorphism and responsive design. |
| **State Management** | Using `st.session_state` to maintain theme preference across app reruns. |
| **Problem Solving** | Handling missing columns gracefully, dynamic filter updates, empty state messages. |
| **Version Control** | Git/GitHub – the project is tracked and shared. |
| **Documentation** | Writing a comprehensive README that explains both usage and development process. |

---

## 📊 What the Dashboard Does – A Tour

### Landing Page
You see the dashboard title and a sidebar with filters. By default, the **dark theme** is active (easy on the eyes).

### Sidebar Filters
- **Date Range** – pick any start/end date. All charts update.
- **Region** – choose one or multiple regions (e.g., West, East).
- **Category** – Technology, Furniture, Office Supplies.
- **Sub‑Category** – dynamically filtered based on category selection.
- **Customer Segment** – Consumer, Corporate, Home Office.
- **Theme Toggle** – switch to light mode instantly.

### KPI Cards (Top)
Six cards show:
- **Sales** – total revenue ($)
- **Profit** – net earnings
- **Margin** – profit / sales (%)
- **Avg Discount** – average discount applied
- **Units Sold** – total quantity
- **Orders** – number of unique orders

### Time Series Chart
Shows monthly Sales (purple line) and Profit (green line). Hover to see exact values.

### Top 5 Customers
Horizontal bar chart – see who spent the most.

### Category Breakdown
Donut pie chart – proportion of sales by category.

### Sub‑Category Treemap
Rectangles sized by sales – perfect for comparing many sub‑categories at once.

### Region & Segment Performance
Bar charts showing sales by region and by customer segment.

### Export & Raw Data
- Download the currently filtered data as CSV.
- Expand a table to inspect the exact rows.

All charts update instantly when you change any filter.

---

## 🚀 How to Run It Yourself

1. **Clone or download** `superstore_dashboard.py`
2. **Install requirements**:
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn plotly
