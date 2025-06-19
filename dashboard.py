import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
import plotly.express as px
from pathlib import Path
import json
import zipfile
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode


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
tab1, tab2, tab3 = st.tabs(["📦 Penjualan Bulanan", "🌍 Penjualan Wilayah", "📊 Statistik Produk"])

# ================= TAB 1: GRAFIK + FILTER =================
with tab1:
    st.title('📦 Penjualan Bulanan')
    col_filter, col_graph = st.columns([1, 3])

    with col_filter:
        # Filter Setting
        st.markdown("### 🔧 Filter")
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
    with col_graph:
        with st.spinner('Loading category sales chart...'):
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
            st.success("Sales chart is ready ✅")
    
            # 🔥 Top 5 Selling Cities
            st.markdown("### 🏙️ Top 5 Cities with Most Sales")
            top_city = selected_data.groupby(['customer_city', 'customer_state']).size().reset_index(name='total_items_sold')
            top_city = top_city.sort_values('total_items_sold', ascending=False).head(5)
            top_city['city_state'] = top_city['customer_city'] + ", " + top_city['customer_state']
    
            gb = GridOptionsBuilder.from_dataframe(top_city[['city_state', 'total_items_sold']])
            gb.configure_selection(selection_mode='single', use_checkbox=True)
            grid_options = gb.build()
    
            grid_response = AgGrid(
                top_city[['city_state', 'total_items_sold']],
                gridOptions=grid_options,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                height=300,
                width='100%',
                theme='alpine'
            )
    
            selected_rows = grid_response['selected_rows']
            if selected_rows:
                selected_row = selected_rows[0]
                selected_city = selected_row['city_state'].split(", ")[0]
                selected_state = selected_row['city_state'].split(", ")[1]
    
                st.markdown(f"#### 📦 Product Details in **{selected_city}, {selected_state}**")
                product_details = selected_data[
                    (selected_data['customer_city'] == selected_city) &
                    (selected_data['customer_state'] == selected_state)
                ]['product_category_name'].value_counts().reset_index()
                product_details.columns = ['Product Category', 'Total Sold']
                st.dataframe(product_details, use_container_width=True)


# ================= TAB 2: PENJUALAN WILAYAH =================
with tab2:
    st.markdown("### 🌍 Penjualan Wilayah")

    geojson_data = load_geojson("br.json")

    # Mapping kode ↔ nama state
    id_to_name = {f["properties"]["id"]: f["properties"]["name"] for f in geojson_data["features"]}
    name_to_id = {v: k for k, v in id_to_name.items()}

    # Layout peta kiri, metrik+tabel kanan
    col_map, col_info = st.columns([3, 2])

    with col_map:
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
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="black",
            plot_bgcolor="black",
            coloraxis_showscale=False,
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
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        # Dropdown nama wilayah
        all_states = sorted(customers_final_dataset['customer_state'].map(id_to_name).dropna().unique())
        selected_state_name = st.selectbox("Pilih negara bagian:", all_states)
        selected_state = name_to_id[selected_state_name]
        st.markdown(f"#### 🧭 Analisis untuk: {selected_state_name}")

        state_filtered = orders_final_dataset[orders_final_dataset['customer_state'] == selected_state]
        top_customers = customers_final_dataset[customers_final_dataset['customer_state'] == selected_state]

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            max_price = state_filtered['price'].max()
            st.metric("💰 Harga Tertinggi", f"R$ {max_price:,.2f}")

        with col_m2:
            max_delivery = state_filtered['delivery_time'].max()
            st.metric("📦 Waktu Delivery Terlama", f"{max_delivery} hari")

        # Tabel Kota
        city_counts = top_customers['customer_city'].value_counts().reset_index()
        city_counts.columns = ['Kota', 'Jumlah Konsumen']
        st.markdown("##### 🏙️ Top 5 Kota")
        st.dataframe(city_counts.head(5), use_container_width=True)

        # Tabel Kategori
        top_categories = state_filtered['product_category_name'].value_counts().reset_index()
        top_categories.columns = ['Kategori Produk', 'Jumlah Terjual']
        st.markdown("##### 🛍️ Top 5 Kategori Produk Terjual")
        st.dataframe(top_categories.head(5), use_container_width=True)



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
                bin_df = df.groupby('price_bin', observed=True).size().reset_index(name='count')
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
