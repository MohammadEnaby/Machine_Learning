import streamlit as st
import plotly.express as px
from PCA import load_data, group_and_aggregate_data, remove_sparse_columns, dimensionality_reduction
#https://pca2025.streamlit.app/
def display_explained_variance(explained_variance_ratio):
    """
    Display the explained variance ratio for each principal component.
    
    Args:
        explained_variance_ratio (list): List of explained variance ratios for each principal component.
    """
    st.write("### Explained Variance Ratio")
    variance_percentages = [f"PC{i+1}: {ratio*100:.2f}%" for i, ratio in enumerate(explained_variance_ratio)]
    st.write("\n".join(variance_percentages))

def plot_reduced_data(reduced_data, num_components, hover_column, title_suffix):
    """
    Create and display scatter plots for dimensionality-reduced data.
    
    Args:
        reduced_data (DataFrame): DataFrame containing the reduced data.
        num_components (int): Number of components for the reduction (2 or 3).
        hover_column (str): Column name for hover data in the plot.
        title_suffix (str): Title suffix for the plot.
    """
    if num_components == 2:
        fig = px.scatter(
            reduced_data,
            x="PC1",
            y="PC2",
            hover_data={hover_column: True},
            title=f"Dimensionality Reduction: {title_suffix} (2D)",
            labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"}
        )
    elif num_components == 3:
        fig = px.scatter_3d(
            reduced_data,
            x="PC1",
            y="PC2",
            z="PC3",
            hover_data={hover_column: True},
            title=f"Dimensionality Reduction: {title_suffix} (3D)",
            labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2", "PC3": "Principal Component 3"}
        )
    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig)

def process_data(df, process_type, group_by_column, agg_func, threshold, num_components):
    """
    Process the dataset based on user-selected options and perform dimensionality reduction.
    
    Args:
        df (DataFrame): The input dataset.
        process_type (str): Processing type ('City-wise' or 'Party-wise').
        group_by_column (str): Column name to group by.
        agg_func (str): Aggregation function ('sum', 'mean', 'count').
        threshold (int): Sparsity threshold to filter columns.
        num_components (int): Number of components for dimensionality reduction.
    """
    if process_type == "City-wise":
        df_grouped = group_and_aggregate_data(df, group_by_column, agg_func)
        df_sparse_removed = remove_sparse_columns(df_grouped, threshold)
        st.write("### Aggregated Data (City-wise)")
        st.write(df_sparse_removed)

        reduced_data, explained_variance_ratio = dimensionality_reduction(df_sparse_removed, num_components, ["city_name"])
        plot_reduced_data(reduced_data, num_components, "city_name", "Cities")
        display_explained_variance(explained_variance_ratio)

    elif process_type == "Party-wise":
        df_grouped = group_and_aggregate_data(df, "city_name", "sum").T
        df_grouped.columns = df_grouped.iloc[0]
        df_grouped = df_grouped[1:].reset_index().rename(columns={"index": "party_name"})
        df_sparse_removed = remove_sparse_columns(df_grouped, threshold)
        st.write("### Aggregated Data (Party-wise)")
        st.write(df_sparse_removed)

        reduced_data, explained_variance_ratio = dimensionality_reduction(df_sparse_removed, num_components, ["party_name"])
        plot_reduced_data(reduced_data, num_components, "party_name", "Parties")
        display_explained_variance(explained_variance_ratio)

# App title
st.title("Dimensionality Reduction App")

# File uploader
st.markdown("### Upload Your Dataset")
uploaded_file = st.file_uploader("Upload CSV or Excel:", type=["csv", "xlsx"], help="Supported formats: CSV, Excel")

if uploaded_file is not None:
    file_path = uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        df = load_data(file_path)
        st.success("File successfully loaded!")
        st.write("### Dataset Preview")
        st.write(df.head(10))

        # Sidebar options for processing
        st.sidebar.markdown("## Processing Options")
        group_by_column = st.sidebar.selectbox("Group By Column:", df.columns)
        agg_func = st.sidebar.selectbox("Aggregation Function:", ["sum", "mean", "count"])
        num_components = st.sidebar.slider("Number of Components:", 2, 3, 2)
        threshold = st.sidebar.number_input("Sparsity Threshold:", min_value=0, value=1000, step=10)
        process_type = st.sidebar.radio("Processing Type:", ["City-wise", "Party-wise"], help="Choose the data grouping type.")

        # Dynamic updates based on user input
        with st.spinner("Processing data..."):
            process_data(df, process_type, group_by_column, agg_func, threshold, num_components)

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a dataset to get started.")
