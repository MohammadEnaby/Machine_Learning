import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def load_data(file_path):
    ext = os.path.splitext(file_path)[1]
    if ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext == '.xlsx':
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    df['city_name'] = df['city_name'].apply(lambda x: x[::-1])
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
    # Separate metadata and numeric columns
    metadata = df[meta_columns]
    numeric_data = df.drop(columns=meta_columns)
    
    # Standardize the numeric data
    standardized_data = (numeric_data - numeric_data.mean()) / numeric_data.std()
    
    # Check for NaN or infinite values and handle them
    if standardized_data.isnull().values.any():
        standardized_data = standardized_data.fillna(0)  # Option to fill NaN with 0 or another strategy
    if (standardized_data == np.inf).values.any() or (standardized_data == -np.inf).values.any():
        standardized_data = standardized_data.replace([np.inf, -np.inf], 0)  # Replace infinity with 0
    
    # Transpose the data (ensure it is 2D)
    standardized_data_T = standardized_data.T
    
    # Verify data type and ensure all values are numeric
    if not np.issubdtype(standardized_data_T.dtypes[0], np.number):
        standardized_data_T = standardized_data_T.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, replace errors with NaN
        standardized_data_T = standardized_data_T.fillna(0)  # Replace NaN with zero

    # Calculate covariance matrix
    try:
        cov_matrix = np.cov(standardized_data_T)  # This should now be a 2D array
    except Exception as e:
        print(f"Error in covariance matrix calculation: {e}")
        return None, None

    # Eigenvalue and eigenvector calculation
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvectors = eigenvectors.real
    eigenvalues = eigenvalues.real
    
    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    
    # Select top components
    top_components = eigenvectors_sorted[:, :num_components]
    
    # Project the data onto the top components
    reduced_data = np.dot(standardized_data, top_components)
    
    # Calculate variance explained by each principal component
    total_variance = np.sum(eigenvalues_sorted)
    explained_variance_ratio = eigenvalues_sorted[:num_components] / total_variance

    # Create DataFrame for the reduced data
    df_reduced = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(num_components)])

    # Add variance ratio to the DataFrame
    explained_variance_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(num_components)],
        'Explained Variance Ratio': (explained_variance_ratio * 100).round(2).astype(str) + '%'
    })

    # Combine metadata with reduced data
    final_df = pd.concat([metadata.reset_index(drop=True), df_reduced], axis=1)

    return final_df, explained_variance_df


def compare_cities(file_path, threshold):
    
    df = load_data(file_path)

    # Group and aggregate data
    df = group_and_aggregate_data(df, 'city_name', 'sum')

    # Remove sparse columns
    df = remove_sparse_columns(df, threshold)

    # Perform dimensionality reduction
    df_reduced, explained_variance_df = dimensionality_reduction(df, num_components=2, meta_columns=['city_name'])


    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df_reduced["PC1"], df_reduced["PC2"])
    for i, city in enumerate(df_reduced["city_name"]):
        plt.text(df_reduced["PC1"][i], df_reduced["PC2"][i], city, fontsize=5)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Dimensionality Reduction: Cities")
    plt.show()
    
        # Display the resulting DataFrame
    print(df_reduced)

    # Display the explained variance ratio
    print("\nExplained Variance Ratio:")
    print(explained_variance_df)


def compare_parties(file_path, threshold):
    df = load_data(file_path)
    
    # Group by city_name and sum the values
    df = group_and_aggregate_data(df, 'city_name', 'sum')

    # organize data for parties view
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
    df_reduced, explained_variance_df = dimensionality_reduction(df_transposed, num_components=2, meta_columns=['party_name'])
    # Display the resulting DataFrame
    print(df_reduced)

    # Display the explained variance ratio
    print("\nExplained Variance Ratio:")
    print(explained_variance_df)
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    # Plot only the real part if the data contains complex numbers
    plt.scatter(df_reduced["PC1"], df_reduced["PC2"])
    for i, party in enumerate(df_reduced["party_name"]):
        plt.text(df_reduced["PC1"][i], df_reduced["PC2"][i], party, fontsize=6)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Dimensionality Reduction: Parties")
    plt.show()


compare_parties("ex1/knesset_25.xlsx", 1000)
