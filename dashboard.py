import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
import plotly.express as px
from pathlib import Path
import json

# Streamlit Config
st.set_page_config(page_title="Dashboard Penjualan", page_icon="📊", layout="wide")

# Caching Functions
@st.cache_data
def load_data():
    orders = pd.read_csv("data/orders_dataset.csv")
    products = pd.read_csv("data/products_dataset.csv")
    order_items = pd.read_csv("data/order_items_dataset.csv")
    customers = pd.read_csv("data/customers_dataset.csv")

    merged = pd.merge(order_items, products, on="product_id")
    merged = pd.merge(merged, orders, on="order_id")
    final = pd.merge(merged, customers, on="customer_id")

    return final, customers

@st.cache_resource
def load_geojson(file_path):
    with open(file_path) as f:
        return json.load(f)

with st.spinner("📦 Loading dataset..."):
    final_dataset, customer_dataset = load_data()
st.success("✅ Data berhasil dimuat")

# Outlier Detection & Cleaning
q1 = final_dataset['price'].quantile(0.25)
q3 = final_dataset['price'].quantile(0.75)
iqr = q3 - q1
lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr

final_dataset['order_delivered_carrier_date'] = pd.to_datetime(final_dataset['order_delivered_carrier_date'])
final_dataset['order_estimated_delivery_date'] = pd.to_datetime(final_dataset['order_estimated_delivery_date'])
final_dataset['order_delivered_customer_date'] = pd.to_datetime(final_dataset['order_delivered_customer_date'], errors='coerce')
rata_rata = (final_dataset['order_delivered_customer_date'] - final_dataset['order_delivered_carrier_date']).mean()
final_dataset['order_delivered_customer_date'] = final_dataset['order_delivered_customer_date'].fillna(final_dataset['order_delivered_carrier_date'] + rata_rata)
final_dataset.dropna(subset=['order_delivered_customer_date'], inplace=True)

final_dataset['delivery_time'] = (final_dataset['order_delivered_customer_date'] - final_dataset['order_delivered_carrier_date']).dt.days
final_dataset = final_dataset[final_dataset['delivery_time'] >= 0]

final_dataset['year'] = final_dataset['order_delivered_customer_date'].dt.year
final_dataset['month'] = final_dataset['order_delivered_customer_date'].dt.month
final_dataset['product_category_name'] = final_dataset['product_category_name'].fillna('Unknown')
final_dataset = final_dataset[(final_dataset['price'] >= lower_bound) & (final_dataset['price'] <= upper_bound)]

# Drop unnecessary columns
final_dataset.drop(columns=[
    'shipping_limit_date', 'product_name_lenght', 'product_description_lenght',
    'product_photos_qty', 'product_weight_g', 'product_length_cm',
    'product_height_cm', 'product_width_cm', 'order_approved_at',
    'order_delivered_carrier_date', 'order_estimated_delivery_date'
], inplace=True)

# Reorder Columns
final_dataset = final_dataset[[
    'order_id', 'order_item_id','product_id', 'customer_id', 'seller_id','price', 
    'product_category_name', 'order_status', 'order_delivered_customer_date', 'delivery_time', 'year', 'month',
    'customer_unique_id', 'customer_city', 'customer_state'
]]

st.info("✅ Data cleaning selesai")

# Sidebar Filter
st.sidebar.title('📊 Filter Setting')
year_list = sorted(final_dataset['year'].dropna().unique(), reverse=True)
selected_year = st.sidebar.selectbox('Select a year', year_list, index=year_list.index(2018))

month_range = st.sidebar.slider('Select month range', 1, 12, (1, 12))
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agust', 'Sept', 'Okt', 'Nov', 'Des']
selected_labels = month_labels[month_range[0]-1:month_range[1]]

category = st.sidebar.multiselect(
    "Category Product (Max 4)",
    options=sorted(final_dataset['product_category_name'].unique()),
    default=['cama_mesa_banho', 'beleza_saude', 'esporte_lazer', 'moveis_decoracao']
)
if len(category) > 4:
    st.sidebar.error("Pilih maksimal 4 kategori saja.")
    st.stop()

# Filter Data
filtered = final_dataset[
    (final_dataset['year'] == selected_year) &
    (final_dataset['month'].between(month_range[0], month_range[1])) &
    (final_dataset['order_status'] == 'delivered') &
    (final_dataset['product_category_name'].isin(category))
].copy()

# Tabs
tab1, tab2, tab3 = st.tabs(["📦 Items per Category", "🌍 Distribusi Konsumen", "📈 Lainnya"])

with tab1:
    st.markdown("### Items Sold Per Categories")
    monthly = filtered.groupby(['month', 'product_category_name']).size().reset_index(name='item_count')
    monthly['month'] = monthly['month'].apply(lambda x: month_labels[x-1])
    monthly['month'] = pd.Categorical(monthly['month'], categories=month_labels, ordered=True)
    pivot = monthly.pivot(index='month', columns='product_category_name', values='item_count').fillna(0)

    colors = ['#8bc091', '#4a998f', '#2c7e8c', '#1c6187', '#28417a']
    color_map = colors[:len(category)] + [colors[-1]] * (len(category) - len(colors))
    st.bar_chart(pivot, color=color_map)
    st.success("✅ Grafik berhasil ditampilkan")

with tab2:
    st.markdown("### Consumer Distribution Map")
    with st.spinner("🔍 Memuat peta..."):
        geojson_data = load_geojson('br.json')

        # Agregasi unik customer per state
        customer_unique = customer_dataset.drop_duplicates('customer_id')
        state_counts = customer_unique.groupby('customer_state').size().reset_index(name='count')
        state_counts['state_name'] = state_counts['customer_state'].map({
            feature['properties']['id']: feature['properties']['name']
            for feature in geojson_data['features']
        })

        # Plotly Choropleth
        fig = px.choropleth(
            state_counts,
            geojson=geojson_data,
            locations='customer_state',
            featureidkey="properties.id",
            color='count',
            color_continuous_scale='Tealgrn_r',  # versi reversed dari 'Tealgrn' agar warna gelap = padat
            range_color=(0, state_counts['count'].max()),
            labels={'count': 'Jumlah Konsumen'},
            hover_name='state_name'
        )

        fig.update_geos(
            visible=False,
            showcountries=False,
            showsubunits=True,
            showframe=False,
            bgcolor='black',
            resolution=110,
            showland=True,
            landcolor='black',
            lakecolor='black',
        )

        fig.update_traces(marker_line_width=0.5, marker_line_color="white")

        fig.update_layout(
            paper_bgcolor='black',
            geo_bgcolor='black',
            geo=dict(scope='south america', center={"lat": -14.2350, "lon": -51.9253}, projection_scale=2.3),
            height=500,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ Peta berhasil ditampilkan")


with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Top 10 Kategori")
        category_count = final_dataset['product_category_name'].value_counts().reset_index()
        category_count.columns = ['product_category_name', 'total_items_sold']
        st.table(category_count.head(10))

    with col2:
        st.markdown("### Distribusi Harga")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_alpha(0)
        ax.hist(final_dataset['price'], bins=30, edgecolor='white', color='#4a998f')
        ax.set_title('Distribusi Harga', color='white')
        ax.set_xlabel('Harga', color='white')
        ax.set_ylabel('Frekuensi', color='white')
        ax.tick_params(axis='both', colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        st.pyplot(fig)
