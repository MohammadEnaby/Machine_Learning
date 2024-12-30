# Importing the required libraries
import streamlit as st  # Streamlit for building interactive web applications
import plotly.express as px  # Plotly for creating interactive plots

# Importing custom functions from the data_loader module
from data_loader import load_data, group_and_aggregate_data, remove_sparse_columns, dimensionality_reduction

# Setting the title of the Streamlit app
st.title("Dimensionality Reduction App")

# Creating an uploader widget for the user to upload a CSV or Excel file
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel):", type=["csv", "xlsx"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Saving the uploaded file locally
    file_path = uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # Write the file content to a local file

    # Loading the data from the uploaded file using the load_data function
    df = load_data(file_path)
    
    # Displaying the first few rows of the dataset as a preview
    st.write("### Dataset Preview")
    st.write(df.head())

    # Creating a dropdown to select a column to group the data by
    group_by_column = st.selectbox("Select a column to group by:", df.columns)

    # Creating a dropdown to select an aggregation function (sum, mean, max, min)
    agg_func = st.selectbox("Select an aggregation function:", ["sum", "mean", "max", "min"])

    # Creating a slider to select the number of components for dimensionality reduction
    num_components = st.slider("Number of components for dimensionality reduction:", min_value=1, max_value=5, value=2)

    # Creating a radio button to select the type of processing (City-wise or Party-wise)
    process_type = st.radio("Choose processing type:", ["City-wise", "Party-wise"])

    # Creating a number input to set the sparsity threshold
    threshold = st.number_input("Set sparsity threshold:", min_value=0, value=1000)

    # Processing the data when the "Process Data" button is clicked
    if st.button("Process Data"):
        # City-wise processing
        if process_type == "City-wise":
            # Grouping and aggregating the data by the selected column and aggregation function
            df_grouped = group_and_aggregate_data(df, group_by_column, agg_func)

            # Removing sparse columns based on the specified threshold
            df_sparse_removed = remove_sparse_columns(df_grouped, threshold)

            # Performing dimensionality reduction (PCA) on the sparse-removed data
            reduced_data = dimensionality_reduction(df_sparse_removed, num_components, ["city_name"])

            # Displaying the reduced dataset
            st.write("### Reduced Dataset")
            st.write(reduced_data)

            # Displaying an interactive scatter plot of the reduced data
            st.write("### Interactive Visualization")
            fig = px.scatter(
                reduced_data,  # Data to plot
                x="PC1",  # X-axis as Principal Component 1
                y="PC2",  # Y-axis as Principal Component 2
                hover_data={"city_name": True},  # Display city names on hover
                title="Dimensionality Reduction: Cities",  # Plot title
                labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"}  # Axis labels
            )
            st.plotly_chart(fig)  # Render the plotly chart

        # Party-wise processing
        elif process_type == "Party-wise":
            # Grouping and aggregating the data by city_name (sum of values)
            df_grouped = group_and_aggregate_data(df, "city_name", "sum").T

            # Setting the first row as column headers and resetting the index
            df_grouped.columns = df_grouped.iloc[0]
            df_grouped = df_grouped[1:].reset_index().rename(columns={"index": "party_name"})

            # Removing sparse columns based on the specified threshold
            df_sparse_removed = remove_sparse_columns(df_grouped, threshold)

            # Performing dimensionality reduction (PCA) on the sparse-removed data
            reduced_data = dimensionality_reduction(df_sparse_removed, num_components, ["party_name"])

            # Displaying the reduced dataset
            st.write("### Reduced Dataset")
            st.write(reduced_data)

            # Displaying an interactive scatter plot of the reduced data
            st.write("### Interactive Visualization")
            fig = px.scatter(
                reduced_data,  # Data to plot
                x="PC1",  # X-axis as Principal Component 1
                y="PC2",  # Y-axis as Principal Component 2
                hover_data={"party_name": True},  # Display party names on hover
                title="Dimensionality Reduction: Parties",  # Plot title
                labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"}  # Axis labels
            )
            st.plotly_chart(fig)  # Render the plotly chart
