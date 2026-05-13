import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ── Page config
st.set_page_config(
    page_title="E-Commerce Intelligence Platform",
    page_icon="🛒",
    layout="wide"
)

# ── Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: #1A1D2E;
        border: 1px solid #6C63FF;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #6C63FF;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #AAAAAA;
    }
    .section-header {
        border-left: 4px solid #6C63FF;
        padding-left: 10px;
        margin: 20px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data
@st.cache_data
def load_data():
    df = pd.read_csv('retail_clean.csv.gz', compression='gzip')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

@st.cache_data
def load_rfm():
    return pd.read_csv('rfm_segmented.csv')

@st.cache_data
def load_purchase_matrix():
    return pd.read_csv('purchase_matrix.csv', index_col=0)

@st.cache_resource
def load_model():
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# ── Load everything
with st.spinner('Loading data...'):
    df       = load_data()
    rfm      = load_rfm()
    pm       = load_purchase_matrix()
    xgb, scaler = load_model()

# ── Product descriptions lookup
product_desc = (df[['StockCode','Description']]
                .drop_duplicates()
                .set_index('StockCode'))

# ── Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=60)
st.sidebar.title("E-Commerce Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🔮 Churn Predictor",
    "👥 Customer Segments",
    "🎯 Recommendations",
    "💡 Key Insights"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Online Retail II (UCI)")
st.sidebar.markdown("**Period:** Dec 2009 – Dec 2011")
st.sidebar.markdown("**Records:** 824,293 transactions")

# ════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Business Overview")
    st.markdown("Real-time analytics across 2 years of UK e-commerce transactions.")
    st.markdown("---")

    # ── KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    total_revenue  = df[df['IsCancelled']==False]['TotalAmount'].sum()
    total_orders   = df[df['IsCancelled']==False]['Invoice'].nunique()
    total_customers= df['CustomerID'].nunique()
    avg_order_val  = total_revenue / total_orders

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">£{total_revenue/1e6:.2f}M</div>
            <div class="metric-label">Total Revenue</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_orders:,}</div>
            <div class="metric-label">Total Orders</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_customers:,}</div>
            <div class="metric-label">Unique Customers</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">£{avg_order_val:.2f}</div>
            <div class="metric-label">Avg Order Value</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Monthly Revenue Trend
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header"><b>Monthly Revenue Trend</b></div>',
                    unsafe_allow_html=True)
        monthly = (df[df['IsCancelled']==False]
                   .groupby(df['InvoiceDate'].dt.to_period('M'))['TotalAmount']
                   .sum().reset_index())
        monthly['InvoiceDate'] = monthly['InvoiceDate'].astype(str)
        fig = px.area(monthly, x='InvoiceDate', y='TotalAmount',
                      color_discrete_sequence=['#6C63FF'])
        fig.update_layout(
            plot_bgcolor='#1A1D2E', paper_bgcolor='#1A1D2E',
            font_color='#FAFAFA', showlegend=False,
            xaxis_title='Month', yaxis_title='Revenue (£)',
            margin=dict(l=10, r=10, t=10, b=10)
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header"><b>Revenue by Country (Top 10)</b></div>',
                    unsafe_allow_html=True)
        country_rev = (df[df['IsCancelled']==False]
                       .groupby('Country')['TotalAmount']
                       .sum().sort_values(ascending=False)
                       .head(10).reset_index())
        fig2 = px.bar(country_rev, x='TotalAmount', y='Country',
                      orientation='h',
                      color='TotalAmount',
                      color_continuous_scale='Purples')
        fig2.update_layout(
            plot_bgcolor='#1A1D2E', paper_bgcolor='#1A1D2E',
            font_color='#FAFAFA', showlegend=False,
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Top Products
    st.markdown('<div class="section-header"><b>Top 10 Products by Revenue</b></div>',
                unsafe_allow_html=True)
    top_products = (df[df['IsCancelled']==False]
                    .groupby('Description')['TotalAmount']
                    .sum().sort_values(ascending=False)
                    .head(10).reset_index())
    fig3 = px.bar(top_products, x='Description', y='TotalAmount',
                  color='TotalAmount', color_continuous_scale='Purples')
    fig3.update_layout(
        plot_bgcolor='#1A1D2E', paper_bgcolor='#1A1D2E',
        font_color='#FAFAFA', showlegend=False,
        xaxis_tickangle=30,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════
# PAGE 2 — CHURN PREDICTOR
# ════════════════════════════════════════
elif page == "🔮 Churn Predictor":
    st.title("🔮 Customer Churn Predictor")
    st.markdown("Enter customer behaviour metrics to predict churn probability.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Customer Parameters")
        frequency      = st.slider("Purchase Frequency (orders)", 1, 200, 5)
        monetary       = st.number_input("Total Spend (£)", 10.0, 100000.0, 500.0, step=50.0)
        f_score        = st.slider("Frequency Score (1–5)", 1, 5, 2)
        m_score        = st.slider("Monetary Score (1–5)", 1, 5, 2)
        total_orders   = st.slider("Total Orders", 1, 200, 5)
        total_items    = st.number_input("Total Items Bought", 1, 50000, 100)
        avg_order_val  = st.number_input("Avg Order Value (£)", 1.0, 10000.0, 100.0)
        lifespan_days  = st.slider("Customer Lifespan (days)", 0, 740, 180)

        predict_btn = st.button("🔮 Predict Churn", use_container_width=True)

    with col2:
        st.markdown("### Prediction Result")
        if predict_btn:
            input_data = np.array([[
                frequency, monetary, f_score, m_score,
                total_orders, total_items, avg_order_val, lifespan_days
            ]])

            prob       = xgb.predict_proba(input_data)[0][1]
            prediction = xgb.predict(input_data)[0]

            # ── Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability", 'font': {'color': 'white'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white'},
                    'bar': {'color': '#e74c3c' if prob > 0.5 else '#2ecc71'},
                    'steps': [
                        {'range': [0, 40],  'color': '#1A1D2E'},
                        {'range': [40, 70], 'color': '#2d2d44'},
                        {'range': [70, 100],'color': '#3d1f1f'}
                    ],
                    'threshold': {
                        'line': {'color': 'white', 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                },
                number={'suffix': '%', 'font': {'color': 'white'}}
            ))
            fig.update_layout(
                paper_bgcolor='#1A1D2E',
                font_color='white',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

            if prediction == 1:
                st.error(f"⚠️ High Churn Risk — {prob*100:.1f}% probability")
                st.markdown("""
                **Recommended Actions:**
                - Send personalised re-engagement email
                - Offer loyalty discount on next order
                - Flag for account manager follow-up
                """)
            else:
                st.success(f"✅ Low Churn Risk — {prob*100:.1f}% probability")
                st.markdown("""
                **Recommended Actions:**
                - Enrol in loyalty rewards programme
                - Cross-sell complementary products
                - Maintain regular communication
                """)
        else:
            st.info("👈 Set parameters and click Predict Churn")

            # Show model performance stats
            st.markdown("### Model Performance")
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            with perf_col1:
                st.metric("AUC-ROC", "0.863")
            with perf_col2:
                st.metric("Accuracy", "76%")
            with perf_col3:
                st.metric("Churn Recall", "81%")


# ════════════════════════════════════════
# PAGE 3 — CUSTOMER SEGMENTS
# ════════════════════════════════════════
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segmentation")
    st.markdown("RFM-based K-Means clustering into 4 business segments.")
    st.markdown("---")

    # ── Segment KPIs
    seg_summary = rfm.groupby('Segment').agg(
        Customers    =('CustomerID','count'),
        Avg_Recency  =('Recency','mean'),
        Avg_Frequency=('Frequency','mean'),
        Avg_Monetary =('Monetary','mean')
    ).round(1).reset_index()

    col1, col2, col3, col4 = st.columns(4)
    colors_seg = {
        'VIP Whales':    '#f39c12',
        'Champions':     '#2ecc71',
        'Loyal Regulars':'#3498db',
        'Hibernating':   '#e74c3c'
    }
    for col, (_, row) in zip([col1,col2,col3,col4], seg_summary.iterrows()):
        color = colors_seg.get(row['Segment'], '#6C63FF')
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}">
                <div class="metric-value" style="color:{color}">{int(row['Customers'])}</div>
                <div class="metric-label">{row['Segment']}</div>
                <div class="metric-label">Avg Spend: £{row['Avg_Monetary']:,.0f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header"><b>Segment Distribution</b></div>',
                    unsafe_allow_html=True)
        fig = px.pie(seg_summary, values='Customers', names='Segment',
                     color='Segment',
                     color_discrete_map=colors_seg,
                     hole=0.4)
        fig.update_layout(
            paper_bgcolor='#1A1D2E', font_color='#FAFAFA',
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header"><b>Recency vs Monetary</b></div>',
                    unsafe_allow_html=True)
        fig2 = px.scatter(rfm, x='Recency', y='Monetary',
                          color='Segment',
                          color_discrete_map=colors_seg,
                          opacity=0.6, size_max=8,
                          hover_data=['CustomerID','Frequency'])
        fig2.update_layout(
            plot_bgcolor='#1A1D2E', paper_bgcolor='#1A1D2E',
            font_color='#FAFAFA',
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Segment explorer
    st.markdown('<div class="section-header"><b>Segment Explorer</b></div>',
                unsafe_allow_html=True)
    selected_seg = st.selectbox("Select Segment", rfm['Segment'].unique())
    seg_df = rfm[rfm['Segment'] == selected_seg].sort_values('Monetary', ascending=False)
    st.dataframe(
        seg_df[['CustomerID','Country','Recency','Frequency',
                'Monetary','TotalOrders','AvgOrderValue']].head(20),
        use_container_width=True
    )


# ════════════════════════════════════════
# PAGE 4 — RECOMMENDATIONS
# ════════════════════════════════════════
elif page == "🎯 Recommendations":
    st.title("🎯 Product Recommendation Engine")
    st.markdown("Item-item collaborative filtering on purchase history.")
    st.markdown("---")

    customer_ids = pm.index.astype(str).tolist()
    selected_customer = st.selectbox(
        "Select Customer ID",
        customer_ids,
        index=0
    )

    n_recs = st.slider("Number of Recommendations", 3, 10, 5)
    rec_btn = st.button("🎯 Get Recommendations", use_container_width=False)

    if rec_btn:
        cust_id = float(selected_customer) if '.' in selected_customer else int(selected_customer)

        if cust_id not in pm.index:
            st.error("Customer not found in purchase matrix")
        else:
            bought      = pm.loc[cust_id]
            bought_list = bought[bought > 0].index.tolist()

            if not bought_list:
                st.warning("No purchase history found for this customer")
            else:
                # ── Item similarity
                item_sim = cosine_similarity(pm.T)
                item_sim_df = pd.DataFrame(item_sim, index=pm.columns, columns=pm.columns)

                unbought = bought[bought == 0].index.tolist()
                scores = {}
                for product in unbought:
                    sim_scores = item_sim_df.loc[product, bought_list]
                    scores[product] = sim_scores.mean()

                top_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_recs]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Previously Bought")
                    for sc in bought_list[:8]:
                        desc = product_desc.loc[sc, 'Description'] if sc in product_desc.index else 'Unknown'
                        st.markdown(f"- `{sc}` {desc}")

                with col2:
                    st.markdown("### Recommended Products")
                    rec_data = []
                    for sc, score in top_recs:
                        desc = product_desc.loc[sc, 'Description'] if sc in product_desc.index else 'Unknown'
                        rec_data.append({'Product': desc, 'Score': round(score, 4)})
                        st.markdown(f"- `{sc}` **{desc}** *(score: {score:.3f})*")

                # ── Score bar chart
                st.markdown("### Recommendation Confidence")
                rec_df = pd.DataFrame(rec_data)
                fig = px.bar(rec_df, x='Score', y='Product',
                             orientation='h',
                             color='Score',
                             color_continuous_scale='Purples')
                fig.update_layout(
                    plot_bgcolor='#1A1D2E', paper_bgcolor='#1A1D2E',
                    font_color='#FAFAFA',
                    yaxis={'categoryorder': 'total ascending'},
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════
# PAGE 5 — KEY INSIGHTS
# ════════════════════════════════════════
elif page == "💡 Key Insights":
    st.title("💡 Key Business Insights")
    st.markdown("Data-driven findings from 2 years of transaction analysis.")
    st.markdown("---")

    # ── HuggingFace Sentiment Section
    st.markdown('<div class="section-header"><b>🤗 Product Sentiment Analysis — HuggingFace RoBERTa</b></div>',
                unsafe_allow_html=True)
    st.caption("Model: cardiffnlp/twitter-roberta-base-sentiment | 5,331 unique products analysed")

    @st.cache_data
    def load_sentiment():
        return pd.read_csv('product_sentiment.csv')

    sentiment_df = load_sentiment()

    col1, col2 = st.columns(2)

    with col1:
        sent_counts = sentiment_df['Sentiment'].value_counts().reset_index()
        sent_counts.columns = ['Sentiment', 'Count']
        colors = {'Positive': '#2ecc71', 'Neutral': '#3498db', 'Negative': '#e74c3c'}
        fig = px.pie(sent_counts, values='Count', names='Sentiment',
                     color='Sentiment',
                     color_discrete_map=colors,
                     hole=0.4,
                     title='Sentiment Distribution')
        fig.update_layout(
            paper_bgcolor='#1A1D2E', font_color='#FAFAFA',
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**🟢 Most Positive Products**")
        top_pos = (sentiment_df[sentiment_df['Sentiment'] == 'Positive']
                   .nlargest(5, 'Confidence')[['Description', 'Confidence']])
        for _, row in top_pos.iterrows():
            st.markdown(f"🟢 `{str(row['Description'])[:45]}` — **{row['Confidence']:.3f}**")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**🔴 Most Negative Products**")
        top_neg = (sentiment_df[sentiment_df['Sentiment'] == 'Negative']
                   .nlargest(5, 'Confidence')[['Description', 'Confidence']])
        for _, row in top_neg.iterrows():
            st.markdown(f"🔴 `{str(row['Description'])[:45]}` — **{row['Confidence']:.3f}**")

    # ── Confidence distribution
    st.markdown('<div class="section-header"><b>Model Confidence Distribution</b></div>',
                unsafe_allow_html=True)
    fig2 = px.histogram(sentiment_df, x='Confidence', color='Sentiment',
                        color_discrete_map=colors,
                        nbins=50, barmode='overlay', opacity=0.7,
                        title='Confidence Score Distribution by Sentiment')
    fig2.update_layout(
        plot_bgcolor='#1A1D2E', paper_bgcolor='#1A1D2E',
        font_color='#FAFAFA',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Key Business Insight Cards
    st.markdown('<div class="section-header"><b>📌 Key Business Insights</b></div>',
                unsafe_allow_html=True)

    insight_data = [
        ("🐋", "VIP Whales",
         "4 customers account for disproportionate revenue averaging £436K each. "
         "Losing a single Whale = £436K revenue impact."),
        ("📉", "Revenue Leakage",
         "£916,724 lost to UK cancellations alone across 6,959 orders. "
         "Fragile product categories (glass, ceramic) dominate cancellations."),
        ("📅", "Seasonal Peak",
         "November 2010 generated £1.17M — nearly double any other month. "
         "Q4 accounts for ~35% of annual revenue."),
        ("😴", "Hibernating Opportunity",
         "2,012 customers haven't purchased in 462 days on average. "
         "Win-back campaigns could recover ~£1.5M in dormant revenue."),
        ("🌍", "International Value",
         "Netherlands customer (#2) spent £528K — international accounts "
         "punch above their weight vs volume."),
        ("🔁", "Retention Pattern",
         "Dec 2009 cohort dropped from 955 to 337 in month 1, "
         "but stabilised at 300–400 — wholesale buyers on irregular schedules."),
        ("🏆", "Top Product",
         "REGENCY CAKESTAND 3 TIER generated £286K across 3,317 orders "
         "and 1,314 unique customers — highest revenue and widest reach."),
        ("🤖", "Churn Signal",
         "F_Score (purchase frequency pattern) is 5x more predictive of churn "
         "than monetary value alone per SHAP analysis.")
    ]

    col1, col2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(insight_data):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:15px; text-align:left">
                <div style="font-size:1.5rem">{icon} <b>{title}</b></div>
                <div style="color:#CCCCCC; margin-top:8px">{desc}</div>
            </div>""", unsafe_allow_html=True)
