import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
import plotly.express as px
from pathlib import Path
import json
import zipfile

# Streamlit Configuration
st.set_page_config(
    page_title="Dashboard Penjualan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ekstrak ZIP geolocation_dataset.zip
geo_zip_path = Path("data/geolocation_dataset.zip")
geo_extract_path = Path("data")
geo_csv_path = geo_extract_path / "geolocation_dataset.csv"

if not geo_csv_path.exists() and geo_zip_path.exists():
    with zipfile.ZipFile(geo_zip_path, 'r') as zip_ref:
        zip_ref.extractall(geo_extract_path)

# Load Datasets
@st.cache_data
def load_data():
    base_dir = Path(".")  # relatif dari file dashboard.py
    orders = pd.read_csv(base_dir / "data" / "orders_dataset.csv")
    products = pd.read_csv(base_dir / "data" / "products_dataset.csv")
    items = pd.read_csv(base_dir / "data" / "order_items_dataset.csv")
    customers = pd.read_csv(base_dir / "data" / "customers_dataset.csv")
    geo = pd.read_csv(base_dir / "data" / "geolocation_dataset.csv")

    # Bersihkan data geolocation: rata-rata koordinat tiap prefix
    geo_clean = geo.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()
    customers['customer_zip_code_prefix'] = customers['customer_zip_code_prefix'].astype(int)
    customers_final = pd.merge(customers, geo_clean, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    customers_final.dropna(inplace=True)

    # Merge orders dengan items dan products
    merged = items.merge(products, on="product_id").merge(orders, on="order_id").merge(customers, on="customer_id")

    return merged, customers_final

# Load data
orders_final_dataset, customers_final_dataset = load_data()

# Pembersihan dan fitur turunan seperti sebelumnya
orders_final_dataset['order_delivered_carrier_date'] = pd.to_datetime(orders_final_dataset['order_delivered_carrier_date'])
orders_final_dataset['order_estimated_delivery_date'] = pd.to_datetime(orders_final_dataset['order_estimated_delivery_date'])
orders_final_dataset['order_delivered_customer_date'] = pd.to_datetime(orders_final_dataset['order_delivered_customer_date'], errors='coerce')
avg_delivery = (orders_final_dataset['order_delivered_customer_date'] - orders_final_dataset['order_delivered_carrier_date']).mean()
orders_final_dataset['order_delivered_customer_date'] = orders_final_dataset['order_delivered_customer_date'].fillna(orders_final_dataset['order_delivered_carrier_date'] + avg_delivery)
orders_final_dataset.dropna(subset=['order_delivered_customer_date'], inplace=True)
orders_final_dataset['delivery_time'] = (orders_final_dataset['order_delivered_customer_date'] - orders_final_dataset['order_delivered_carrier_date']).dt.days
orders_final_dataset = orders_final_dataset[orders_final_dataset['delivery_time'] >= 0]
orders_final_dataset['year'] = orders_final_dataset['order_delivered_customer_date'].dt.year
orders_final_dataset['month'] = orders_final_dataset['order_delivered_customer_date'].dt.month
orders_final_dataset['product_category_name'] = orders_final_dataset['product_category_name'].fillna('Unknown')

# Buat kolom warna hitam peta
@st.cache_data
def load_geojson(file_path):
    with open(file_path) as f:
        return json.load(f)
# START DATA VISUALIZATION IN STREAMLIT

# Tab layout
st.title('E-Commerce Public Dashboard')
tab1, tab2, tab3 = st.tabs(["📦 Penjualan Bulanan", "🌍 Peta Konsumen", "📊 Statistik Produk"])

# ================= SIDEBAR =================
with st.sidebar:
    st.title('📊 Filter Setting')
    orders_final_dataset = orders_final_dataset.dropna(subset=['year'])
    orders_final_dataset['year'] = orders_final_dataset['year'].astype(int)
    year_list = sorted(orders_final_dataset['year'].unique(), reverse=True)
    selected_year = st.selectbox('Select a year', year_list, index=year_list.index(2018))

    values = st.slider('Select month range', 1, 12, (1, 12))
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agust', 'Sept', 'Okt', 'Nov', 'Des']
    selected_labels = month_labels[values[0]-1:values[1]]

    category = st.multiselect(
        label="Category Product (Max 3)",
        options=sorted(orders_final_dataset['product_category_name'].unique()),
        default=['cama_mesa_banho', 'beleza_saude', 'esporte_lazer', 'moveis_decoracao']
    )
    if len(category) > 4:
        st.error("You can select up to 4 categories only. Please deselect some options.")
        st.stop()

# ================= TAB 1: GRAFIK =================
with tab1:
    st.markdown('### 📦 Items Sold Per Categories')
    selected_data = orders_final_dataset[
        (orders_final_dataset['year'] == selected_year) &
        (orders_final_dataset['month'].between(values[0], values[1])) &
        (orders_final_dataset['order_status'] == 'delivered') &
        (orders_final_dataset['product_category_name'].isin(category))
    ]
    monthly_selected_category = selected_data.groupby(['month', 'product_category_name']).size().reset_index(name='item_count')
    monthly_selected_category['month'] = monthly_selected_category['month'].apply(lambda x: month_labels[x-1])
    monthly_selected_category['month'] = pd.Categorical(monthly_selected_category['month'], categories=month_labels, ordered=True)
    monthly_selected_category = monthly_selected_category.sort_values('month')
    chart_data = monthly_selected_category.pivot(index='month', columns='product_category_name', values='item_count').fillna(0)

    colors = ['#8bc091', '#4a998f', '#2c7e8c', '#1c6187', '#28417a']
    color_map = colors[:len(category)] + [colors[-1]] * (len(category) - len(colors))
    st.bar_chart(chart_data, color=color_map)

# ================= TAB 2: MAP =================
with tab2:
    st.markdown("### 🌍 Peta Sebaran Konsumen")

    @st.cache_resource
    def load_geojson(file_path):
        with open(file_path) as f:
            return json.load(f)

    geojson_data = load_geojson("br.json")
    state_counts = customers_final_dataset.groupby("customer_state").size().reset_index(name="count")
    state_counts['state_name'] = state_counts['customer_state'].map({
        feature["id"]: feature["properties"]["name"] for feature in geojson_data["features"]
    })

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
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="black"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="black",
        plot_bgcolor="black",
        geo=dict(
            showframe=False,
            showcoastlines=False,
            showland=True,
            landcolor='black',
            countrycolor='white',
            subunitcolor='white'
        ),
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3: TOP KATEGORI & HARGA =================
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('### 🔝 Top 10 Kategori')
        category_count = orders_final_dataset['product_category_name'].value_counts().reset_index()
        category_count.columns = ['product_category_name', 'total_items_sold']
        top_categories = category_count.head(10)
        st.table(top_categories)
    
    with col2:
        st.markdown('### 💸 Distribusi Harga')
        data = orders_final_dataset['price'].dropna()
    
        if data.empty:
            st.warning("Data harga tidak tersedia.")
        else:
            # Hapus outlier dengan metode IQR
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
            if filtered_data.empty:
                st.warning("Tidak ada data harga yang tersisa setelah menghapus outlier.")
            else:
                # Gunakan Freedman–Diaconis rule untuk bin size
                q25, q75 = filtered_data.quantile([0.25, 0.75])
                bin_width = 2 * (q75 - q25) * len(filtered_data) ** (-1/3)
                bin_size = round(bin_width, 2) if bin_width > 0 else 10
    
                fig = px.histogram(
                    filtered_data,
                    x=filtered_data,
                    nbins=int((filtered_data.max() - filtered_data.min()) / bin_size),
                    title="Distribusi Harga (tanpa Outlier)",
                    color_discrete_sequence=["#4a998f"]
                )
                fig.update_layout(
                    xaxis_title="Harga",
                    yaxis_title="Frekuensi",
                    font=dict(color="white"),
                    plot_bgcolor="black",
                    paper_bgcolor="black"
                )
                st.plotly_chart(fig, use_container_width=True)
