import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# LOAD DATA
# ==========================

@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

@st.cache_resource
def load_model():
    with open("kmeans_model.pkl","rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl","rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_model()

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html,body,[class*="css"]{
font-family:'Poppins',sans-serif;
background:#F8FAFC;
}

/* Hide Streamlit */

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

/* Main */

.block-container{
padding-top:2rem;
padding-left:2.5rem;
padding-right:2.5rem;
}

/* Sidebar */

section[data-testid="stSidebar"]{
background:#0F172A;
}

section[data-testid="stSidebar"] *{
color:white;
}

/* Radio */

div[role="radiogroup"] label{
padding:10px;
margin-bottom:8px;
border-radius:12px;
}

div[role="radiogroup"] label:hover{
background:#1E3A8A;
}

/* Cards */

.card{
background:white;
padding:25px;
border-radius:20px;
box-shadow:0 8px 18px rgba(0,0,0,.08);
}

/* Hero */

.hero{
background:linear-gradient(135deg,#2563EB,#3B82F6);
padding:45px;
border-radius:25px;
color:white;
box-shadow:0 15px 30px rgba(37,99,235,.25);
}

</style>
""",unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================

st.sidebar.markdown(
"""
# 🛍️ Customer Segmentation

---

### Navigation
"""
)

page = st.sidebar.radio(
"",
[
"🏠 Home",
"📊 Dashboard",
"🎯 Predict",
"ℹ️ About"
]
)

st.sidebar.markdown("---")

st.sidebar.caption("Machine Learning Project")

st.sidebar.caption("Developed by Disha Goyal ")

# ==========================
# TEMP PAGE
# ==========================

# ==========================
# HOME PAGE
# ==========================

if page == "🏠 Home":

    left, right = st.columns([1.4,1])

    with left:

        st.markdown("""
        <div class='hero'>

        <h1 style='font-size:46px;margin-bottom:10px;'>
        Customer Segmentation Dashboard
        </h1>

        <p style='font-size:18px;line-height:1.7;opacity:.95;'>

        Discover meaningful customer groups using Machine Learning.
        Analyze customer behaviour based on Age, Annual Income and
        Spending Score to support smarter marketing decisions.

        </p>

        </div>
        """,unsafe_allow_html=True)

    with right:
        st.write("")
        st.write("")

    c1,c2,c3,c4 = st.columns(4)

    with c1:

        st.markdown(f"""
        <div class='card'>

        <h2 style='margin-bottom:0;'>👥 {len(df)}</h2>

        <span>Total Customers</span>

        </div>
        """,unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class='card'>

        <h2 style='margin-bottom:0;'>👤 {round(df['Age'].mean())}</h2>

        <span>Average Age</span>

        </div>
        """,unsafe_allow_html=True)

    with c3:

        st.markdown(f"""
        <div class='card'>

        <h2 style='margin-bottom:0;'>💰 {round(df['Annual Income (k$)'].mean())}k</h2>

        <span>Average Income</span>

        </div>
        """,unsafe_allow_html=True)

    with c4:

        st.markdown(f"""
        <div class='card'>

        <h2 style='margin-bottom:0;'>⭐ {round(df['Spending Score (1-100)'].mean())}</h2>

        <span>Spending Score</span>

        </div>
        """,unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.subheader("Project Overview")

    col1,col2 = st.columns(2)

    with col1:

        st.success("Customer Segmentation using K-Means Clustering")

        st.success("Interactive Analytics Dashboard")

        st.success("Customer Cluster Prediction")

    with col2:

        st.info("Business Insights")

        st.info("Marketing Strategy Support")

        st.info("Easy Data Visualization")

    st.write("")

    st.markdown("---")

    st.caption(
        "This project demonstrates how unsupervised machine learning can help businesses understand customer behaviour and design personalized marketing strategies."
    )
# ==========================
# DASHBOARD
# ==========================

elif page == "📊 Dashboard":

    st.title("📊 Customer Dashboard")
    st.caption("Explore customer demographics and spending behaviour.")

    st.write("")

    # -------- Metrics --------

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Customers", len(df))
    m2.metric("Average Age", round(df["Age"].mean(), 1))
    m3.metric("Average Income", f"{round(df['Annual Income (k$)'].mean(),1)} k$")
    m4.metric("Average Spending", round(df["Spending Score (1-100)"].mean(),1))

    st.write("")

    # -------- Charts --------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            df,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            color="Gender",
            title="Income vs Spending Score",
            template="plotly_white",
            height=450
        )

        fig.update_layout(
            title_x=0.2,
            margin=dict(l=20,r=20,t=50,b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.histogram(
            df,
            x="Age",
            nbins=20,
            color="Gender",
            title="Age Distribution",
            template="plotly_white",
            height=450
        )

        fig.update_layout(
            title_x=0.25,
            margin=dict(l=20,r=20,t=50,b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    st.write("")

    col3, col4 = st.columns(2)

    with col3:

        gender = df["Gender"].value_counts().reset_index()

        gender.columns = ["Gender", "Count"]

        fig = px.pie(
            gender,
            names="Gender",
            values="Count",
            hole=0.55,
            title="Gender Distribution",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col4:

        fig = px.box(
            df,
            x="Gender",
            y="Annual Income (k$)",
            color="Gender",
            title="Income Distribution",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.write("")

    st.subheader("Dataset Preview")

    st.dataframe(df, use_container_width=True)
# ==========================
# PREDICTION PAGE
# ==========================

elif page == "🎯 Predict":

    st.title("🎯 Customer Segment Prediction")
    st.caption("Predict the customer segment using the trained K-Means model.")

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        age = st.slider(
            "Age",
            18,
            70,
            30
        )

    with col2:

        income = st.slider(
            "Annual Income (k$)",
            15,
            140,
            60
        )

        spending = st.slider(
            "Spending Score",
            1,
            100,
            50
        )

    st.write("")

    if st.button("Predict Customer Segment"):

        gender_value = 1 if gender == "Male" else 0

        data = np.array([[gender_value, age, income, spending]])

        data_scaled = scaler.transform(data)

        cluster = model.predict(data_scaled)[0]

        st.success(f"Customer belongs to **Cluster {cluster}**")

        st.write("")

        # ---------------- Recommendations ----------------

        if cluster == 0:

            st.info("""
### 🟢 Cluster 0

**Budget Conscious Customers**

**Recommendation**

- Offer discounts
- Cashback coupons
- Festival offers
- Loyalty rewards
""")

        elif cluster == 1:

            st.success("""
### 🟡 Cluster 1

**High Value Customers**

**Recommendation**

- Premium Membership
- Early Access Products
- VIP Rewards
- Personalized Services
""")

        elif cluster == 2:

            st.warning("""
### 🔵 Cluster 2

**Potential Customers**

**Recommendation**

- Personalized Marketing
- Combo Offers
- Product Recommendations
- Email Campaigns
""")

        elif cluster == 3:

            st.error("""
### 🔴 Cluster 3

**Low Engagement Customers**

**Recommendation**

- Re-engagement Campaigns
- Limited Time Discounts
- Push Notifications
- Referral Offers
""")

        else:

            st.info("""
### 🟣 Cluster 4

**Regular Customers**

**Recommendation**

- Reward Points
- Seasonal Offers
- Membership Discounts
- Product Bundles
""")
# ==========================
# ABOUT PAGE
# ==========================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("""
### 🛍️ Customer Segmentation Dashboard

This project uses **K-Means Clustering** to divide customers into different groups based on their shopping behaviour.

The segmentation is performed using the following customer attributes:

- Age
- Annual Income
- Spending Score

The dashboard allows users to:

- Explore customer data interactively
- Visualize customer behaviour
- Predict customer segments
- Generate business recommendations

---

### 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- Plotly
- Scikit-learn
- K-Means Clustering

---

### 📂 Dataset

Mall Customers Dataset

Total Records: **200 Customers**


### 👩‍💻 Developed By

**Disha Goyal**
    """)