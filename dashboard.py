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
    # Read the CSV file using the relative path
    order_dataset = pd.read_csv("data/orders_dataset.csv", delimiter=",")
    products_dataset = pd.read_csv("data/products_dataset.csv", delimiter=",")
    order_items_dataset = pd.read_csv("data/order_items_dataset.csv", delimiter=",")
    customer_dataset = pd.read_csv("data/customers_dataset.csv", delimiter=",")

    # Merge datasets
    merged = pd.merge(
        left=order_items_dataset,
        right=products_dataset,
        how="inner",
        left_on="product_id",
        right_on="product_id"
    )
    orders_dataset = pd.merge(
        left=merged,
        right=order_dataset,
        how="inner",
        left_on="order_id",
        right_on="order_id"
    )
    final_dataset = pd.merge(
        left=orders_dataset,
        right=customer_dataset,
        how="inner",
        left_on="customer_id",
        right_on="customer_id"
    )

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
    
    return final_dataset

final_dataset = load_data() 
customer_dataset=  load_data()


# START EXPLORATORY DATA ANALYSIS

# 1. Display statistics of final_dataset
print(final_dataset.describe())

# START DATA VISUALIZATION IN STREAMLIT

st.title('E-Commerce Public Dashboard')
# 1. Populate the sidebar with filter options 
with st.sidebar:
    st.title('📊 Filter Setting')
    final_dataset = final_dataset.dropna(subset=['year'])
    final_dataset['year'] = final_dataset['year'].astype(int)
    year_list = sorted(final_dataset['year'].unique(), reverse=True)
    selected_year = st.selectbox('Select a year', year_list, index=year_list.index(2018))
    
    values = st.slider(
        label='Select month range',
        min_value=1, max_value=12, value=(1, 12))
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agust', 'Sept', 'Okt', 'Nov', 'Des']
    selected_labels = month_labels[values[0]-1:values[1]]
    
    category = st.multiselect(
        label="Category Product (Max 3)",
        options=sorted(final_dataset['product_category_name'].unique()),
        default=['cama_mesa_banho', 'beleza_saude', 'esporte_lazer', 'moveis_decoracao']
    )
    if len(category) > 4:
        st.error("You can select up to 4 categories only. Please deselect some options.")
        st.stop()

# 2. Divide into 3 columns       
col = st.columns((5, 5), gap='large')

with col[0]:
    
    # 3. Add a bar chart to visualize the total items sold in each category and each month
    selected_data = final_dataset[(final_dataset['year'] == selected_year) &
                                (final_dataset['month'].between(values[0], values[1])) & 
                                (final_dataset['order_status'] == 'delivered') &
                                (final_dataset['product_category_name'].isin(category))]

    # Group and format data
    monthly_selected_category = selected_data.groupby(['month', 'product_category_name']).size().reset_index(name='item_count')

    # Map month numbers to month_labels and sort
    monthly_selected_category['month'] = monthly_selected_category['month'].apply(lambda x: month_labels[x-1])
    month_order = month_labels  # Use month_labels for sorting
    monthly_selected_category['month'] = pd.Categorical(monthly_selected_category['month'], categories=month_order, ordered=True)
    monthly_selected_category = monthly_selected_category.sort_values('month')

    # Create DataFrame for plotting
    chart_data = monthly_selected_category.pivot_table(index='month', columns='product_category_name', values='item_count', fill_value=0)

    # Display chart
    st.markdown('#### Items Sold Per Categories')
    st.bar_chart(chart_data, color=['#691f55', '#b81656', '#ed503e', '#f5a47b')
    
    # 4. Create a consumer distribution map for E-Commerce
    def make_choropleth(dataset_name, state_id, count_column, color_theme, geojson_file):

        with open(geojson_file) as f:
            geojson_data = json.load(f)
            
        state_counts['state_name'] = state_counts['customer_state'].map(
        {feature['properties']['id']: feature['properties']['name'] for feature in geojson_data['features']}
    )
        choropleth = px.choropleth(
            dataset_name,
            geojson=geojson_data,
            locations=state_id,
            featureidkey="properties.id",
            color=count_column,
            color_continuous_scale=color_theme,
            range_color=(0, dataset_name[count_column].max()), 
            labels={count_column: 'Customers'},
            hover_name="state_name"
        )
        choropleth.update_layout(
            geo=dict(
                scope='south america',
                center={"lat": -14.2350, "lon": -51.9253},
                projection_scale=2,
                showland=True,
                landcolor='black',
                oceancolor='black',
                bgcolor='black',
                countrycolor='black',
                subunitcolor='black'
            ),
            paper_bgcolor='black',
            plot_bgcolor='black',
            template='plotly_dark',
            margin=dict(l=0, r=0, t=0, b=0),
            height=500,
            width=500
        )
        return choropleth
    
    unique_customers = customer_dataset.drop_duplicates(subset='customer_id', keep='first')
    state_counts = unique_customers.groupby('customer_state').size().reset_index(name='count')
    choropleth_map = make_choropleth(state_counts, 'customer_state', 'count', 'matter', 'br.json')
    st.markdown('#### Consumer Distribution Map')
    st.plotly_chart(choropleth_map)

with col[1]:
    
    # Hitung kategori teratas
    category_count = final_dataset['product_category_name'].value_counts().reset_index()
    category_count.columns = ['product_category_name', 'total_items_sold']
    top_categories = category_count.head(10)

    # Ubah angka menjadi string
    top_categories['total_items_sold'] = top_categories['total_items_sold'].astype(str)

    # Tampilkan tabel dengan st.table
    st.markdown('#### Top 10 Kategori')
    st.table(top_categories)
    
    # 6. Create a pie chart showing the proportion of sales by price range
    st.markdown('#### Price Distribution')
    data = final_dataset['price']
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    palette = sns.color_palette('rocket', n_colors=10)
    n, bins, patches = ax.hist(data, bins=30, edgecolor='white', color='#98137D')
    ax.set_title('Distribusi Harga', color='white')
    ax.set_xlabel('Harga', color='white')
    ax.set_ylabel('Frekuensi', color='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    ax.tick_params(axis='both', colors='white')
    for patch in patches:
        patch.set_edgecolor('white')
    plt.tight_layout()
    st.pyplot(fig)
