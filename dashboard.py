import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
import plotly.express as px
from pathlib import Path
import json

# Streamlit Configuration
st.set_page_config(
    page_title="Dashboard Penjualan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ekstrak data for CSV files
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

final_dataset, customer_dataset = load_data()

# Outlier Identification with IQR
q1 = final_dataset['price'].quantile(0.25)
q3 = final_dataset['price'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers_condition = (final_dataset['price'] < lower_bound) | (final_dataset['price'] > upper_bound)

print("Outliers harga:")
print(sum(outliers_condition))

# Display dataset information
print("===== DATA INFO =====")
print(final_dataset.info())
print(customer_dataset.info())

# Count missing values in each column
print("\n===== MISSING VALUES =====")
print(final_dataset.isnull().sum())

# Display descriptive statistics to identify potential outliers
print("\n===== DESCRIPTIVE STATISTICS =====")
print(final_dataset.describe())


# START DATA CLEANING

# 1. Convert date data type to datetime
final_dataset['order_delivered_carrier_date'] = pd.to_datetime(final_dataset['order_delivered_carrier_date'])
final_dataset['order_estimated_delivery_date'] = pd.to_datetime(final_dataset['order_estimated_delivery_date'])
final_dataset['order_delivered_customer_date'] = pd.to_datetime(final_dataset['order_delivered_customer_date'], errors='coerce')

# 2. Fill missing data in 'order_delivered_customer_date' with the average difference between 
#    'order_delivered_customer_date' and 'order_delivered_carrier_date'
average_delivery_time = (final_dataset['order_delivered_customer_date'] - final_dataset['order_delivered_carrier_date']).mean()
final_dataset['order_delivered_customer_date'] = final_dataset['order_delivered_customer_date'].fillna(final_dataset['order_delivered_carrier_date'] + average_delivery_time)
final_dataset = final_dataset.dropna(subset=['order_delivered_customer_date'])

# 3. Add a 'delivery_time' column
final_dataset['delivery_time']= (final_dataset['order_delivered_customer_date'] - final_dataset['order_delivered_carrier_date'])
final_dataset['delivery_time'] = final_dataset['delivery_time'].dt.days
final_dataset = final_dataset[final_dataset['delivery_time'] >= 0]

# 4. Add 'year' and 'month' columns based on 'order_delivered_customer_date'
final_dataset['year'] = final_dataset['order_delivered_customer_date'].dt.year
final_dataset['month'] = final_dataset['order_delivered_customer_date'].dt.month

# 5. Fill missing data in the 'product_category_name' column with 'Unknown'
final_dataset.update(final_dataset[['product_category_name']].fillna('Unknown'))

# 6. Remove outliers in the 'price' column
q1 = final_dataset['price'].quantile(0.25)
q3 = final_dataset['price'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers_condition = (final_dataset['price'] < lower_bound) | (final_dataset['price'] > upper_bound)
final_dataset.drop(index=final_dataset[outliers_condition].index, inplace=True)

# 7. Trim unused columns
drop_columns = [
    'shipping_limit_date', 'product_name_lenght',
    'product_description_lenght', 'product_photos_qty', 'product_weight_g',
    'product_length_cm', 'product_height_cm', 'product_width_cm',
    'order_approved_at', 'order_delivered_carrier_date', 'order_estimated_delivery_date'
]
final_dataset.drop(columns=drop_columns, inplace=True)
# 8. Arrange columns to be more structured
columns_order = [
    'order_id', 'order_item_id','product_id', 'customer_id', 'seller_id','price', 
    'product_category_name', 'order_status', 'order_delivered_customer_date', 'delivery_time', 'year', 'month',
    'customer_unique_id', 'customer_city', 'customer_state'
]
final_dataset = final_dataset[columns_order]
    

# START EXPLORATORY DATA ANALYSIS

# 2. Average Delivery Time by Category
average_delivery_time_by_category = final_dataset.groupby('product_category_name')['delivery_time'].mean().sort_values()

# 3. Average Price by Category
average_price_by_category = final_dataset.groupby('product_category_name')['price'].mean().sort_values()

# 4. Total Sales by Year and Month
sales_by_year_month = final_dataset.groupby(['year', 'month'])['price'].sum().unstack()

# 5. Demographic Analysis by City and State
city_distribution = customer_dataset.groupby('customer_city').customer_id.nunique().sort_values(ascending=False)
state_distribution = customer_dataset.groupby('customer_state').customer_id.nunique().sort_values(ascending=False)


# START DATA VISUALIZATION IN STREAMLIT

st.title('E-Commerce Public Dashboard')
# Sidebar Filter
st.sidebar.title('📊 Filter Setting')
year_list = sorted(final_dataset['year'].dropna().unique(), reverse=True)
selected_year = st.sidebar.selectbox('Select a year', year_list, index=year_list.index(2018))
month_range = st.sidebar.slider('Select month range', 1, 12, (1, 12))
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agust', 'Sept', 'Okt', 'Nov', 'Des']
selected_labels = month_labels[month_range[0]-1:month_range[1]]
category = st.sidebar.multiselect("Category Product (Max 4)", 
    options=sorted(final_dataset['product_category_name'].unique()),
    default=['cama_mesa_banho', 'beleza_saude', 'esporte_lazer', 'moveis_decoracao']
)
if len(category) > 4:
    st.sidebar.error("Pilih maksimal 4 kategori saja.")
    st.stop()

filtered = final_dataset[
    (final_dataset['year'] == selected_year) &
    (final_dataset['month'].between(month_range[0], month_range[1])) &
    (final_dataset['order_status'] == 'delivered') &
    (final_dataset['product_category_name'].isin(category))
].copy()

# ======== TABS START ==========
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

with tab2:
    st.markdown("### Consumer Distribution Map")
    geojson_data = load_geojson("br.json")

    unique_customers = customer_dataset.drop_duplicates(subset='customer_id')
    state_counts = unique_customers.groupby('customer_state').size().reset_index(name='count')
    state_name_mapping = {f["id"]: f["properties"]["name"] for f in geojson_data["features"]}
    state_counts["state_name"] = state_counts["customer_state"].map(state_name_mapping)

    fig = px.choropleth(
        state_counts,
        geojson=geojson_data,
        locations="customer_state",
        featureidkey="id",
        color="count",
        color_continuous_scale="Tealgrn",
        hover_name="state_name",
        range_color=(0, state_counts["count"].max()),
        labels={"count": "Customers"}
    )
    fig.update_layout(
        geo=dict(
            scope='south america',
            center={"lat": -14.2350, "lon": -51.9253},
            projection_scale=2,
            showland=True,
            landcolor="black",
            oceancolor="black",
            bgcolor="black",
            countrycolor="white",
            subunitcolor="white"
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        template="plotly_dark",
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig)

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
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.hist(final_dataset['price'], bins=30, edgecolor='white', color='#4a998f')
        ax.set_title('Distribusi Harga', color='white')
        ax.set_xlabel('Harga', color='white')
        ax.set_ylabel('Frekuensi', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        st.pyplot(fig)
