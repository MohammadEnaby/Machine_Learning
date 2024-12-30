import pandas as pd
import numpy as np
import os
import plotly.express as px


def load_data(file_path):
    ext = os.path.splitext(file_path)[1]
    if ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext == '.xlsx':
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    if 'ballot_code' in df.columns:
        df = df.drop('ballot_code', axis=1)
    return df

def group_and_aggregate_data(df, group_by_column, agg_func):
    df = df.reset_index(drop=True)  # Reset index if necessary
    df_grouped = df.groupby(group_by_column).agg(agg_func)
    
    # Reset index after groupby to keep 'city_name' as a column
    df_grouped = df_grouped.reset_index()
    return df_grouped

def remove_sparse_columns(df, threshold):
    for column in df.columns:
        if column != 'city_name' and column != 'party_name':  # Skip the 'city_name' column
            total_sum = df[column].sum()
            if total_sum < threshold:
                df = df.drop(column, axis=1)
    return df


def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
    metadata = df[meta_columns]
    numeric_data = df.drop(columns=meta_columns)
    standardized_data = (numeric_data - numeric_data.mean()) / numeric_data.std()

    standardized_data = standardized_data.fillna(0).replace([np.inf, -np.inf], 0)
    standardized_data_T = standardized_data.T

    cov_matrix = np.cov(standardized_data_T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvectors = eigenvectors.real
    eigenvalues = eigenvalues.real

    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_components = eigenvectors[:, sorted_indices[:num_components]]

    reduced_data = np.dot(standardized_data, top_components)

    df_reduced = pd.DataFrame(reduced_data, columns=[f'PC{i + 1}' for i in range(num_components)])
    final_df = pd.concat([metadata.reset_index(drop=True), df_reduced], axis=1)
    return final_df


def compare_cities(file_path, threshold):
    df = load_data(file_path)

    # Group and aggregate data
    df = group_and_aggregate_data(df, 'city_name', 'sum')

    # Remove sparse columns
    df = remove_sparse_columns(df, threshold)

    # Perform dimensionality reduction
    reduced_data = dimensionality_reduction(df, num_components=2, meta_columns=['city_name'])

    # Create scatter plot using Plotly
    fig =  px.scatter(
                reduced_data,
                x="PC1",
                y="PC2",
                hover_data={"city_name": True},
                title="Dimensionality Reduction: Cities",
                labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"}
            )
    fig.show()
    
    # Display the resulting DataFrame
    print(reduced_data)


def compare_parties(file_path, threshold):
    df = load_data(file_path)
    
    # Group by city_name and sum the values
    df = group_and_aggregate_data(df, 'city_name', 'sum')

    # Organize data for parties view
    df_transposed = df.T
    df_transposed.columns = df_transposed.iloc[0]
    df_transposed = df_transposed[1:]
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'party_name'}, inplace=True)
    df_transposed.columns.name = None

    # Remove sparse columns
    df_transposed = remove_sparse_columns(df_transposed, threshold)
    print(df_transposed)

    # Perform dimensionality reduction
    reduced_data = dimensionality_reduction(df_transposed, num_components=2, meta_columns=['party_name'])

    # Display the resulting DataFrame
    print(reduced_data)

    # Create scatter plot using Plotly
    fig = px.scatter(
                reduced_data,
                x="PC1",
                y="PC2",
                hover_data={"party_name": True},
                title="Dimensionality Reduction: Parties",
                labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"}
            )
    fig.show()

compare_cities("ex1/knesset_25.xlsx",1000)
