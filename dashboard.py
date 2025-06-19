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

# Hapus outlier harga dari awal
q1 = orders_final_dataset['price'].quantile(0.25)
q3 = orders_final_dataset['price'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
orders_final_dataset = orders_final_dataset[
    (orders_final_dataset['price'] >= lower_bound) & 
    (orders_final_dataset['price'] <= upper_bound)
]

# Bersihkan missing values pada customers_final_dataset
customers_final_dataset.dropna(inplace=True)

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
    with st.spinner('Memuat grafik penjualan kategori...'):
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
        chart_data = monthly_selected_category.pivot(index='month', columns='product_category_name', values='item_count').fillna(0)

        colors = ['#8bc091', '#4a998f', '#2c7e8c', '#1c6187', '#28417a']
        color_map = colors[:len(category)] + [colors[-1]] * (len(category) - len(colors))
        st.bar_chart(chart_data, color=color_map)
        st.success("Grafik penjualan siap ditampilkan ✅")
        
        st.markdown("#### 📋 Cuplikan Data Order")
        st.dataframe(orders_final_dataset.head(10))
        
# ================= TAB 2: MAP =================
with tab2:
    with st.spinner('Memuat peta sebaran konsumen...'):
        st.markdown("### 🌍 Peta Sebaran Konsumen")

        # Load GeoJSON
        geojson_data = load_geojson("br.json")

        # Buat mapping dari properties.id → properties.name
        id_to_name = {
            feature["properties"]["id"]: feature["properties"]["name"]
            for feature in geojson_data["features"]
            if "id" in feature["properties"] and "name" in feature["properties"]
        }

        # Hitung jumlah konsumen per state
        state_counts = customers_final_dataset.groupby("customer_state").size().reset_index(name="count")

        # Tambahkan nama state untuk hover (jika tidak cocok, fallback ke kode)
        state_counts['state_name'] = state_counts['customer_state'].map(id_to_name).fillna(state_counts['customer_state'])

        # Tampilkan jumlah negara bagian yang cocok
        st.write("📌 Jumlah Negara Bagian Terdeteksi:", len(state_counts))

        # Buat peta choropleth
        fig = px.choropleth(
            state_counts,
            geojson=geojson_data,
            locations="customer_state",
            featureidkey="properties.id",  # <- sesuaikan dengan struktur br.json
            color="count",
            color_continuous_scale="Tealgrn",
            range_color=(0, state_counts["count"].max()),
            hover_name="state_name",
            labels={"count": "Jumlah Konsumen"}
        )

        # Zoom otomatis berdasarkan lokasi data
        fig.update_geos(
            fitbounds="locations",
            visible=False,
            bgcolor="black"
        )

        # Layout tampilan
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
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )

        # Tampilkan peta
        st.plotly_chart(fig, use_container_width=True)
        st.success("Peta berhasil dimuat ✅")

        # Preview data untuk debugging
        st.markdown("#### 📋 Cuplikan Data Konsumen")
        st.dataframe(customers_final_dataset.head(10))

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
    with st.spinner('Menghitung distribusi harga...'):
        st.markdown('### 💸 Distribusi Harga')
        data = orders_final_dataset['price'].dropna()

        if data.empty:
            st.warning("Data harga tidak tersedia.")
        else:
            bin_size = 20
            price_min = int(data.min())
            price_max = int(data.max())
            bin_edges = list(range(price_min, price_max + bin_size, bin_size))

            df = pd.DataFrame({'price': data})
            df['price_bin'] = pd.cut(df['price'], bins=bin_edges)

            # Hitung frekuensi per bin
            bin_df = df.groupby('price_bin').size().reset_index(name='count')
            bin_df['bin_start'] = bin_df['price_bin'].apply(lambda x: int(x.left))
            bin_df['bin_end'] = bin_df['price_bin'].apply(lambda x: int(x.right))
            bin_df['range_label'] = bin_df['bin_start'].astype(str) + ' - ' + bin_df['bin_end'].astype(str)

            chart = alt.Chart(bin_df).mark_bar(
                color='#4a998f',
                stroke='white',
                strokeWidth=1
            ).encode(
                x=alt.X('range_label:N', title='Harga', sort=None),
                y=alt.Y('count:Q', title='Frekuensi'),
                tooltip=['range_label:N', 'count:Q']
            ).properties(
                width='container',
                height=400,
                title='Distribusi Harga (bin size = 20)'
            ).configure_view(
                stroke=None
            ).configure_axis(
                labelColor='white',
                titleColor='white'
            ).configure_title(
                color='white'
            ).configure(background='black')

            st.altair_chart(chart, use_container_width=True)
            st.success("Distribusi harga selesai ✅")
