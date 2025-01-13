import pandas as pd
import numpy as np
import os
import plotly.express as px


def load_data(file_path):
    """
    Load data from a given file path based on its extension (.csv or .xlsx).

    Parameters:
    - file_path: Path to the file to be loaded.

    Returns:
    - A pandas DataFrame containing the data.
    """
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
    """
    Group data by a specific column and aggregate using a given function.

    Parameters:
    - df: The DataFrame to be grouped.
    - group_by_column: The column to group by.
    - agg_func: Aggregation function to apply (e.g., 'sum').

    Returns:
    - Grouped and aggregated DataFrame.
    """
    df = df.reset_index(drop=True)
    df_grouped = df.groupby(group_by_column).agg(agg_func).reset_index()
    return df_grouped


def remove_sparse_columns(df, threshold):
    """
    Remove columns from the DataFrame where the sum of values is below a given threshold.

    Parameters:
    - df: The DataFrame to process.
    - threshold: Minimum sum threshold for columns to be retained.

    Returns:
    - DataFrame with sparse columns removed.
    """
    for column in df.columns:
        if column not in ['city_name', 'party_name']:  # Skip specific columns
            total_sum = df[column].sum()
            if total_sum < threshold:
                df = df.drop(column, axis=1)
    return df


def dimensionality_reduction(df, num_components, meta_columns):
    """
    Perform PCA for dimensionality reduction on numeric data.

    Parameters:
    - df: DataFrame containing the dataset.
    - num_components: Number of principal components to retain.
    - meta_columns: List of columns to retain as metadata.

    Returns:
    - DataFrame with reduced dimensions and metadata, and explained variance ratios.
    """
    metadata = df[meta_columns]
    numeric_data = df.drop(columns=meta_columns)

    standardized_data = (numeric_data - numeric_data.mean()) / numeric_data.std()
    standardized_data = standardized_data.fillna(0).replace([np.inf, -np.inf], 0)
    
    cov_matrix = np.cov(standardized_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvectors = eigenvectors.real
    eigenvalues = eigenvalues.real

    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_components = eigenvectors[:, sorted_indices[:num_components]]
    reduced_data = np.dot(standardized_data, top_components)

    df_reduced = pd.DataFrame(reduced_data, columns=[f'PC{i + 1}' for i in range(num_components)])
    final_df = pd.concat([metadata.reset_index(drop=True), df_reduced], axis=1)

    explained_variance_ratio = eigenvalues[sorted_indices[:num_components]] / np.sum(eigenvalues)
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"PC{i + 1} explains {ratio * 100:.2f}% of the variance")

    return final_df, explained_variance_ratio


def compare_cities(file_path, threshold):
    """
    Analyze and visualize data for cities using dimensionality reduction.

    Parameters:
    - file_path: Path to the file containing data.
    - threshold: Minimum threshold to remove sparse columns.
    """
    df = load_data(file_path)
    df = group_and_aggregate_data(df, 'city_name', 'sum')
    df = remove_sparse_columns(df, threshold)
    reduced_data, explained_variance_ratio = dimensionality_reduction(df, num_components=2, meta_columns=['city_name'])

    fig = px.scatter(
        reduced_data,
        x="PC1",
        y="PC2",
        hover_data={"city_name": True},
        title="Dimensionality Reduction: Cities",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"}
    )
    fig.show()
    print(reduced_data)


def compare_parties(file_path, threshold):
    """
    Analyze and visualize data for parties using dimensionality reduction.

    Parameters:
    - file_path: Path to the file containing data.
    - threshold: Minimum threshold to remove sparse columns.
    """
    df = load_data(file_path)
    df = group_and_aggregate_data(df, 'city_name', 'sum')

    df_transposed = df.T
    df_transposed.columns = df_transposed.iloc[0]
    df_transposed = df_transposed[1:]
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'party_name'}, inplace=True)
    df_transposed.columns.name = None

    df_transposed = remove_sparse_columns(df_transposed, threshold)
    print(df_transposed)

    reduced_data, explained_variance_ratio = dimensionality_reduction(df_transposed, num_components=2, meta_columns=['party_name'])

    fig = px.scatter(
        reduced_data,
        x="PC1",
        y="PC2",
        hover_data={"party_name": True},
        title="Dimensionality Reduction: Parties",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"}
    )
    fig.show()
    print(reduced_data)


# Example call (uncomment to use)
# compare_cities("knesset_25.xlsx", 20000)
