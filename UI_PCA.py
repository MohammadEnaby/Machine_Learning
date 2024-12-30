import streamlit as st
import plotly.express as px
import numpy as np

from data_loader import load_data, group_and_aggregate_data, remove_sparse_columns, dimensionality_reduction

st.title("Dimensionality Reduction App")

uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel):", type=["csv", "xlsx"])

if uploaded_file is not None:
    file_path = uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    df = load_data(file_path)
    st.write("### Dataset Preview")
    st.write(df.head())

    group_by_column = st.selectbox("Select a column to group by:", df.columns)
    agg_func = st.selectbox("Select an aggregation function:", ["sum", "mean", "max", "min"])
    num_components = st.slider("Number of components for dimensionality reduction:", min_value=1, max_value=5, value=2)
    process_type = st.radio("Choose processing type:", ["City-wise", "Party-wise"])
    threshold = st.number_input("Set sparsity threshold:", min_value=0, value=1000)

    if st.button("Process Data"):
        if process_type == "City-wise":
            df_grouped = group_and_aggregate_data(df, group_by_column, agg_func)
            df_sparse_removed = remove_sparse_columns(df_grouped, threshold)
            reduced_data, explained_variance_ratio = dimensionality_reduction(df_sparse_removed, num_components, ["city_name"])

            st.write("### Reduced Dataset")
            st.write(reduced_data)

            st.write("### Interactive Visualization")
            fig = px.scatter(
                reduced_data,
                x="PC1",
                y="PC2",
                hover_data={"city_name": True},
                title="Dimensionality Reduction: Cities",
                labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"}
            )
            st.plotly_chart(fig)

            # Display explained variance ratio as percentages
            st.write('### Explained Variance Ratio')
            variance_percentages = [f"PC{i+1}: {ratio*100:.2f}%" for i, ratio in enumerate(explained_variance_ratio)]
            st.write("\n".join(variance_percentages))

        elif process_type == "Party-wise":
            df_grouped = group_and_aggregate_data(df, "city_name", "sum").T
            df_grouped.columns = df_grouped.iloc[0]
            df_grouped = df_grouped[1:].reset_index().rename(columns={"index": "party_name"})
            df_sparse_removed = remove_sparse_columns(df_grouped, threshold)
            reduced_data, explained_variance_ratio = dimensionality_reduction(df_sparse_removed, num_components, ["party_name"])

            st.write("### Reduced Dataset")
            st.write(reduced_data)

            st.write("### Interactive Visualization")
            fig = px.scatter(
                reduced_data,
                x="PC1",
                y="PC2",
                hover_data={"party_name": True},
                title="Dimensionality Reduction: Parties",
                labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"}
            )
            st.plotly_chart(fig)

            # Display explained variance ratio as percentages
            st.write('### Explained Variance Ratio')
            variance_percentages = [f"PC{i+1}: {ratio*100:.2f}%" for i, ratio in enumerate(explained_variance_ratio)]
            st.write("\n".join(variance_percentages))
