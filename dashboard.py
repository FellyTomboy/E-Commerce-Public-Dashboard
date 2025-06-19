import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
import plotly.express as px
from zipfile import ZipFile
import os

st.set_page_config(
    page_title="Dashboard Penjualan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ekstrak ZIP jika belum diekstrak
@st.cache_resource
def extract_zip_once():
    if not os.path.exists("data/geolocation_dataset.csv"):
        with ZipFile("data/geolocation_dataset.zip", 'r') as zip_ref:
            zip_ref.extractall("data")
    return True

extract_zip_once()

@st.cache_data
def load_data():
    orders = pd.read_csv("data/orders_dataset.csv")
    products = pd.read_csv("data/products_dataset.csv")
    order_items = pd.read_csv("data/order_items_dataset.csv")
    customers = pd.read_csv("data/customers_dataset.csv")
    geolocation = pd.read_csv("data/geolocation_dataset.csv")

    merged = pd.merge(order_items, products, on="product_id")
    merged = pd.merge(merged, orders, on="order_id")
    final_orders = pd.merge(merged, customers, on="customer_id")

    geo_clean = geolocation.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()
    customers['customer_zip_code_prefix'] = customers['customer_zip_code_prefix'].astype(int)
    customers_final = pd.merge(customers, geo_clean, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    customers_final.dropna(inplace=True)

    customers_final = customers_final[['customer_id', 'customer_unique_id', 'customer_zip_code_prefix',
                                       'customer_city', 'customer_state', 'geolocation_lat', 'geolocation_lng']]

    return final_orders, customers_final

orders_final_dataset, customers_final_dataset = load_data()

# Cleaning
orders_final_dataset['order_delivered_carrier_date'] = pd.to_datetime(orders_final_dataset['order_delivered_carrier_date'])
orders_final_dataset['order_estimated_delivery_date'] = pd.to_datetime(orders_final_dataset['order_estimated_delivery_date'])
orders_final_dataset['order_delivered_customer_date'] = pd.to_datetime(orders_final_dataset['order_delivered_customer_date'], errors='coerce')
average_delivery_time = (orders_final_dataset['order_delivered_customer_date'] - orders_final_dataset['order_delivered_carrier_date']).mean()
orders_final_dataset['order_delivered_customer_date'] = orders_final_dataset['order_delivered_customer_date'].fillna(orders_final_dataset['order_delivered_carrier_date'] + average_delivery_time)
orders_final_dataset.dropna(subset=['order_delivered_customer_date'], inplace=True)
orders_final_dataset['delivery_time'] = (orders_final_dataset['order_delivered_customer_date'] - orders_final_dataset['order_delivered_carrier_date']).dt.days
orders_final_dataset = orders_final_dataset[orders_final_dataset['delivery_time'] >= 0]
orders_final_dataset['year'] = orders_final_dataset['order_delivered_customer_date'].dt.year
orders_final_dataset['month'] = orders_final_dataset['order_delivered_customer_date'].dt.month
orders_final_dataset['product_category_name'] = orders_final_dataset['product_category_name'].fillna('Unknown')

# Remove outliers
q1 = orders_final_dataset['price'].quantile(0.25)
q3 = orders_final_dataset['price'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
orders_final_dataset = orders_final_dataset[(orders_final_dataset['price'] >= lower_bound) & (orders_final_dataset['price'] <= upper_bound)]

# Drop unused columns
drop_columns = [
    'shipping_limit_date', 'product_name_lenght', 'product_description_lenght',
    'product_photos_qty', 'product_weight_g', 'product_length_cm',
    'product_height_cm', 'product_width_cm', 'order_approved_at',
    'order_delivered_carrier_date', 'order_estimated_delivery_date'
]
orders_final_dataset.drop(columns=drop_columns, inplace=True)

columns_order = [
    'order_id', 'order_item_id','product_id', 'customer_id', 'seller_id','price', 
    'product_category_name', 'order_status', 'order_delivered_customer_date', 'delivery_time', 'year', 'month',
    'customer_unique_id', 'customer_city', 'customer_state'
]
orders_final_dataset = orders_final_dataset[columns_order]

# Sidebar
st.sidebar.title('📊 Filter Setting')
year_list = sorted(orders_final_dataset['year'].dropna().unique(), reverse=True)
selected_year = st.sidebar.selectbox('Select a year', year_list, index=year_list.index(2018))
month_range = st.sidebar.slider('Select month range', 1, 12, (1, 12))
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agust', 'Sept', 'Okt', 'Nov', 'Des']
category = st.sidebar.multiselect("Category Product (Max 4)", 
    options=sorted(orders_final_dataset['product_category_name'].unique()),
    default=['cama_mesa_banho', 'beleza_saude', 'esporte_lazer', 'moveis_decoracao']
)
if len(category) > 4:
    st.sidebar.error("Pilih maksimal 4 kategori.")
    st.stop()

filtered = orders_final_dataset[
    (orders_final_dataset['year'] == selected_year) &
    (orders_final_dataset['month'].between(month_range[0], month_range[1])) &
    (orders_final_dataset['order_status'] == 'delivered') &
    (orders_final_dataset['product_category_name'].isin(category))
].copy()

# Tabs
tab1, tab2, tab3 = st.tabs(["📦 Items per Category", "🌍 Sebaran Konsumen", "📈 Lainnya"])

with tab1:
    st.markdown("### Items Sold Per Categories")
    monthly = filtered.groupby(['month', 'product_category_name']).size().reset_index(name='item_count')
    monthly['month'] = monthly['month'].apply(lambda x: month_labels[x-1])
    monthly['month'] = pd.Categorical(monthly['month'], categories=month_labels, ordered=True)
    pivot = monthly.pivot(index='month', columns='product_category_name', values='item_count').fillna(0)
    colors = ['#8bc091', '#4a998f', '#2c7e8c', '#1c6187', '#28417a']
    color_map = colors[:len(category)] + [colors[-1]] * (len(category) - len(colors))
    st.bar_chart(pivot, color=color_map)

with tab2:
    st.markdown("### Persebaran Konsumen")
    fig = px.scatter_geo(
        customers_final_dataset,
        lat='geolocation_lat',
        lon='geolocation_lng',
        hover_name='customer_city',
        hover_data={'customer_state': True},
        color_discrete_sequence=['#4a998f'],
        template='plotly_dark',
        opacity=0.6,
    )
    fig.update_layout(
        geo=dict(
            scope='south america',
            showland=True,
            landcolor='black',
            bgcolor='black',
            countrycolor='white',
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Top 10 Kategori")
        category_count = orders_final_dataset['product_category_name'].value_counts().reset_index()
        category_count.columns = ['product_category_name', 'total_items_sold']
        st.table(category_count.head(10))
    with col2:
        st.markdown("### Distribusi Harga")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.hist(orders_final_dataset['price'], bins=30, edgecolor='white', color='#4a998f')
        ax.set_title('Distribusi Harga', color='white')
        ax.set_xlabel('Harga', color='white')
        ax.set_ylabel('Frekuensi', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        st.pyplot(fig)
