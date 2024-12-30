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


import pandas as pd
import numpy as np

def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
    """
    Perform Principal Component Analysis (PCA) to reduce the dimensionality of the given dataframe.

    Parameters:
    - df: DataFrame containing the dataset.
    - num_components: The number of principal components to retain.
    - meta_columns: List of columns to retain as metadata (not part of PCA).

    Returns:
    - DataFrame with the reduced dimensions and the original metadata.
    """

    # Extract metadata columns (non-numeric data that we don't want to include in PCA)
    metadata = df[meta_columns]

    # Separate numeric data from metadata
    numeric_data = df.drop(columns=meta_columns)

    # Standardize the numeric data: subtract the mean and divide by the standard deviation
    # This ensures that each feature has a mean of 0 and a standard deviation of 1
    standardized_data = (numeric_data - numeric_data.mean()) / numeric_data.std()

    # Replace missing values (NaN) and infinity values with 0
    standardized_data = standardized_data.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Transpose the standardized data for covariance matrix calculation
    standardized_data_T = standardized_data.T

    # Covariance matrix calculation: This matrix captures the covariance (relationship) between features
    cov_matrix = np.cov(standardized_data_T)

    # Eigenvalue decomposition: Extract the eigenvalues and eigenvectors of the covariance matrix
    # Eigenvalues represent the variance explained by each principal component
    # Eigenvectors represent the direction of each principal component in the feature space
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Ensure the eigenvalues and eigenvectors are real numbers (they may be complex due to numerical precision)
    eigenvectors = eigenvectors.real
    eigenvalues = eigenvalues.real

    # Sort eigenvalues in descending order (larger eigenvalues capture more variance)
    # Sort the eigenvectors based on their corresponding eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_components = eigenvectors[:, sorted_indices[:num_components]]

    # Project the original standardized data onto the top principal components
    reduced_data = np.dot(standardized_data, top_components)

    # Convert the reduced data into a DataFrame with appropriate column names (PC1, PC2, etc.)
    df_reduced = pd.DataFrame(reduced_data, columns=[f'PC{i + 1}' for i in range(num_components)])

    # Concatenate the metadata and the reduced data
    final_df = pd.concat([metadata.reset_index(drop=True), df_reduced], axis=1)

    # Explained Variance Ratio: How much variance each principal component explains
    explained_variance_ratio = eigenvalues[sorted_indices[:num_components]] / np.sum(eigenvalues)

    # Print out the explained variance ratios
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"PC{i + 1} explains {ratio * 100:.2f}% of the variance")

    return final_df, explained_variance_ratio



def compare_cities(file_path, threshold):
    df = load_data(file_path)

    # Group and aggregate data
    df = group_and_aggregate_data(df, 'city_name', 'sum')

    # Remove sparse columns
    df = remove_sparse_columns(df, threshold)

    # Perform dimensionality reduction
    reduced_data, explained_variance_ratio = dimensionality_reduction(df, num_components=2, meta_columns=['city_name'])

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
    reduced_data, explained_variance_ratio = dimensionality_reduction(df_transposed, num_components=2, meta_columns=['party_name'])

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
