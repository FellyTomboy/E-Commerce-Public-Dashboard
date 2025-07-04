# E-Commerce Dashboard dengan struktur sesuai permintaan dan kotak tiap komponen
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

st.markdown("""
    <h1 style='font-size: 45px;'>📊 E-Commerce Public Dashboard</h1>
""", unsafe_allow_html=True)

col_wilayah, col_bulanan = st.columns([1, 1])

with col_wilayah:
    with st.container(border=True):
        st.markdown("""
            <h2 style='font-size: 35px;text-align: center;'>Regional Sales</h2>
        """, unsafe_allow_html=True)
        selected_state_name = st.selectbox("Select State:", sorted(customers_final_dataset['customer_state'].map(id_to_name).dropna().unique()))
        selected_state = name_to_id[selected_state_name]
        state_data = orders_final_dataset[orders_final_dataset['customer_state'] == selected_state]
        col_peta, col_keterangan = st.columns([1, 2])

        with col_peta:
            with st.container(border=True):
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
                    labels={"count": "Customer Count"}
                )
                fig.update_geos(fitbounds="locations", visible=False, bgcolor="black")
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="black",
                    plot_bgcolor="black",
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=250,
                    width=300
                )
                st.plotly_chart(fig, use_container_width=False)

            with st.container(border=True):
                state_data['weekday'] = state_data['order_purchase_timestamp'].dt.day_name()
                top_day = state_data['weekday'].value_counts().idxmax()
                st.metric("📆 Busiest Day", top_day)

        with col_keterangan:
            with st.container(border=True):
                col1, col2 = st.columns(2)
                with col1:
                    avg_price = state_data['price'].mean()
                    st.metric("🛒 Avg. Price", f"R$ {avg_price:.2f}")
                with col2:
                    avg_delivery = state_data['delivery_time'].mean()
                    st.metric("⏱️ Avg. Delivery", f"{avg_delivery:.1f} days")

            with st.container(border=True):
                st.markdown("<h5>🏙️ Top 5 Cities by Customers</h5>", unsafe_allow_html=True)
                top5_cities = state_data['customer_city'].value_counts().head(5).reset_index()
                top5_cities.columns = ['City', 'Customer Count']
                if st.toggle("Show Chart", key="chart_cities_customers"):
                    chart = alt.Chart(top5_cities).mark_bar().encode(
                        x='Customer Count:Q',
                        y=alt.Y('City:N', sort='-x'),
                        color=alt.value('#4A998F')
                    ).properties(width='container')
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.dataframe(top5_cities, use_container_width=True, hide_index=True)

        with st.container(border=True):
            st.markdown("<h5>🛍️ Top 5 Sold Products</h5>", unsafe_allow_html=True)
            top_products = state_data['product_category_name'].value_counts().head(5).reset_index()
            top_products.columns = ['Product', 'Items Sold']
            if st.toggle("Show Chart", key="chart_top_products"):
                chart = alt.Chart(top_products).mark_bar().encode(
                    x='Items Sold:Q',
                    y=alt.Y('Product:N', sort='-x'),
                    color=alt.value('#4A998F')
                ).properties(width='container')
                st.altair_chart(chart, use_container_width=True)

            else:
                st.dataframe(top_products, use_container_width=True, hide_index=True)

with col_bulanan:
    with st.container(border=True):
        st.markdown("""
            <h2 style='font-size: 35px;text-align: center;'>Monthly Sales</h2>
        """, unsafe_allow_html=True)
        col_filter, col_grafik = st.columns([1, 2])

        with col_filter:
            with st.container(border=True):
                selected_year = st.selectbox('Year:', sorted(orders_final_dataset['year'].unique(), reverse=True))
                selected_months = st.slider('Month Range', 1, 12, (1, 12))
                month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                selected_categories = st.multiselect(
                    "Product Categories (max. 4):",
                    sorted(orders_final_dataset['product_category_name'].unique()),
                    default=['cama_mesa_banho', 'beleza_saude'],
                    max_selections=4
                )

                filtered = orders_final_dataset[
                    (orders_final_dataset['year'] == selected_year) &
                    (orders_final_dataset['month'].between(*selected_months)) &
                    (orders_final_dataset['order_status'] == 'delivered') &
                    (orders_final_dataset['product_category_name'].isin(selected_categories))
                ]

            with st.container(border=True):
                top_cities = filtered.groupby('customer_city').size().sort_values(ascending=False).head(5).reset_index()
                top_cities.columns = ['City', 'Total Items Sold']
                if st.toggle("Show Chart", key="chart_top_sales"):
                    chart = alt.Chart(top_cities).mark_bar().encode(
                        x='Total Items Sold:Q',
                        y=alt.Y('City:N', sort='-x'),
                        color=alt.value('#4A998F')
                    ).properties(width='container')
                    st.altair_chart(chart, use_container_width=True)

                else:
                    st.dataframe(top_cities, use_container_width=True, hide_index=True)

        with col_grafik:
            with st.container(border=True):
                st.markdown("<h5>📦 Items Sold per Category</h5>", unsafe_allow_html=True)
                monthly_selected_category = filtered.groupby(['month', 'product_category_name']).size().reset_index(name='item_count')
                monthly_selected_category['month'] = monthly_selected_category['month'].apply(lambda x: month_labels[x-1])
                monthly_selected_category['month'] = pd.Categorical(monthly_selected_category['month'], categories=month_labels, ordered=True)
                chart_data = monthly_selected_category.pivot(index='month', columns='product_category_name', values='item_count').fillna(0)

                colors = ['#8bc091', '#4a998f', '#2c7e8c', '#1c6187', '#28417a']
                color_map = colors[:len(selected_categories)] + [colors[-1]] * (len(selected_categories) - len(colors))
                st.bar_chart(chart_data, use_container_width=True, color=color_map)

            with st.container(border=True):
                col1, col2 = st.columns(2)
                with col1:
                    avg_price = filtered['price'].mean()
                    st.metric("🛒 Avg. Price", f"R$ {avg_price:.2f}")
                with col2:
                    avg_delivery = filtered['delivery_time'].mean()
                    st.metric("⏱️ Avg. Delivery", f"{avg_delivery:.1f} days")

            selected_city = st.selectbox("Select City to view product details:", top_cities['City'])

        with st.container(border=True):
            st.markdown(f"<h5>📦 Product Details – {selected_city}</h5>", unsafe_allow_html=True)
            detail = filtered[filtered['customer_city'] == selected_city]['product_category_name'].value_counts().reset_index()
            detail.columns = ['Category', 'Items Sold']
            if st.toggle("Show Chart", key="chart_detail_products"):
                chart = alt.Chart(detail).mark_bar().encode(
                    x='Items Sold:Q',
                    y=alt.Y('Category:N', sort='-x'),
                    color=alt.value('#4A998F')
                ).properties(width='container')
                st.altair_chart(chart, use_container_width=True)
            else:
                st.dataframe(detail, use_container_width=True, hide_index=True)
