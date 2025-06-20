# E-Commerce Dashboard dengan struktur sesuai permintaan
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
import json
from pathlib import Path
import zipfile

st.set_page_config(
    page_title="E-Commerce Public Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data():
    base = Path("data")
    with zipfile.ZipFile(base / "geolocation_dataset.zip", 'r') as zip_ref:
        zip_ref.extractall(base)

    orders = pd.read_csv(base / "orders_dataset.csv")
    products = pd.read_csv(base / "products_dataset.csv")
    items = pd.read_csv(base / "order_items_dataset.csv")
    customers = pd.read_csv(base / "customers_dataset.csv")
    geo = pd.read_csv(base / "geolocation_dataset.csv")

    geo_clean = geo.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()
    customers['customer_zip_code_prefix'] = customers['customer_zip_code_prefix'].astype(int)
    customers_final = pd.merge(customers, geo_clean, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left').dropna()

    merged = items.merge(products, on="product_id").merge(orders, on="order_id").merge(customers, on="customer_id")
    merged['order_delivered_customer_date'] = pd.to_datetime(merged['order_delivered_customer_date'], errors='coerce')
    merged['order_delivered_carrier_date'] = pd.to_datetime(merged['order_delivered_carrier_date'])
    merged['order_purchase_timestamp'] = pd.to_datetime(merged['order_purchase_timestamp'])
    merged['delivery_time'] = (merged['order_delivered_customer_date'] - merged['order_delivered_carrier_date']).dt.days
    merged = merged[merged['delivery_time'] >= 0].dropna()
    merged['year'] = merged['order_delivered_customer_date'].dt.year
    merged['month'] = merged['order_delivered_customer_date'].dt.month
    merged['product_category_name'] = merged['product_category_name'].fillna('Unknown')

    return merged, customers_final

orders_final_dataset, customers_final_dataset = load_data()

@st.cache_data
def load_geojson(file_path):
    with open(file_path) as f:
        return json.load(f)

geojson_data = load_geojson("br.json")
id_to_name = {f["properties"]["id"]: f["properties"]["name"] for f in geojson_data["features"]}
name_to_id = {v: k for k, v in id_to_name.items()}

st.title("📊 E-Commerce Public Dashboard")

col_wilayah, col_bulanan = st.columns([2, 3])

with col_wilayah:
    col_peta = st.container()
    with col_peta:
        state_counts = customers_final_dataset.groupby("customer_state").size().reset_index(name="count")
        state_counts['state_name'] = state_counts['customer_state'].map(id_to_name)
        fig = px.choropleth(
            state_counts,
            geojson=geojson_data,
            locations="customer_state",
            featureidkey="properties.id",
            color="count",
            color_continuous_scale="Tealgrn",
            range_color=(0, state_counts["count"].max()),
            hover_name="state_name",
            labels={"count": "Jumlah Konsumen"}
        )
        fig.update_geos(fitbounds="locations", visible=False, bgcolor="black")
        fig.update_layout(template="plotly_dark", paper_bgcolor="black", plot_bgcolor="black")
        st.plotly_chart(fig, use_container_width=True)

    col_negarabagian = st.container()
    with col_negarabagian:
        selected_state_name = st.selectbox("Pilih Negara Bagian:", sorted(customers_final_dataset['customer_state'].map(id_to_name).dropna().unique()))
        selected_state = name_to_id[selected_state_name]
        state_data = orders_final_dataset[orders_final_dataset['customer_state'] == selected_state]

        col_pembeliantermahal, col_pengirimanterlama = st.columns(2)
        with col_pembeliantermahal:
            st.metric("💰 Pembelian Termahal", f"R$ {state_data['price'].max():,.2f}")
        with col_pengirimanterlama:
            st.metric("📦 Pengiriman Terlama", f"{state_data['delivery_time'].max()} hari")

        top5_cities = state_data['customer_city'].value_counts().head(5).reset_index()
        top5_cities.columns = ['Kota', 'Jumlah Konsumen']
        st.markdown("#### 🏙️ 5 Kota Konsumen Terbanyak")
        st.dataframe(top5_cities, use_container_width=True)

    top_products = state_data['product_category_name'].value_counts().head(5).reset_index()
    top_products.columns = ['Produk', 'Jumlah Terjual']
    st.markdown("#### 🛍️ 5 Produk Terjual Terbanyak")
    st.dataframe(top_products, use_container_width=True)

with col_bulanan:
    col_filter, col_grafikmetrik = st.columns([1, 2])

    with col_filter:
        selected_year = st.selectbox('Tahun:', sorted(orders_final_dataset['year'].unique(), reverse=True))
        selected_months = st.slider('Rentang Bulan', 1, 12, (1, 12))
        selected_categories = st.multiselect(
            "Kategori Produk (maks. 4):",
            sorted(orders_final_dataset['product_category_name'].unique()),
            default=['cama_mesa_banho', 'beleza_saude'],
            max_selections=4
        )

        filtered = orders_final_dataset[
            (orders_final_dataset['year'] == selected_year) &
            (orders_final_dataset['month'].between(*selected_months)) &
            (orders_final_dataset['product_category_name'].isin(selected_categories))
        ]

        top_cities = filtered.groupby('customer_city').size().sort_values(ascending=False).head(5).reset_index()
        top_cities.columns = ['City', 'Total Items Sold']
        st.markdown("### 🏙️ Top 5 Cities by Total Sales")
        st.dataframe(top_cities, use_container_width=True)

    with col_grafikmetrik:
        st.markdown("### 📦 Items Sold per Categories")
        summary = filtered.groupby(['month', 'product_category_name']).size().reset_index(name='count')
        summary['month'] = summary['month'].astype(str)
        fig = px.bar(summary, x='month', y='count', color='product_category_name', barmode='group',
                     template='plotly_dark', labels={'count': 'Items Sold', 'month': 'Month'})
        fig.update_layout(paper_bgcolor="black", plot_bgcolor="black")
        st.plotly_chart(fig, use_container_width=True)

        col_pembeliantermahal, col_pengirimanterlama = st.columns(2)
        with col_pembeliantermahal:
            st.metric("💰 Pembelian Termahal", f"R$ {filtered['price'].max():,.2f}")
        with col_pengirimanterlama:
            st.metric("📦 Pengiriman Terlama", f"{filtered['delivery_time'].max()} hari")

        selected_city = st.selectbox("Pilih Kota untuk lihat detail produk:", top_cities['City'])

    st.markdown(f"### 📦 Detail Produk - {selected_city}")
    detail = filtered[filtered['customer_city'] == selected_city]['product_category_name'].value_counts().reset_index()
    detail.columns = ['Kategori', 'Jumlah Terjual']
    st.dataframe(detail, use_container_width=True)
