# Principal Component Analysis Project

This project contains Python scripts and a Streamlit web application for analyzing and visualizing data related to cities and political parties. The data is loaded from either a CSV or Excel file, processed, and subjected to dimensionality reduction using Principal Component Analysis (PCA). The goal is to identify patterns, relationships, and visualize the data in lower dimensions.

## PCA.py - Python Script

The `PCA.py` script is a standalone Python script for performing data analysis and visualization directly through the command line or IDE. It contains utility functions for loading data, grouping, aggregating, and performing PCA. The script generates scatter plots to visualize the principal components of the reduced dataset.

### Key Features of `PCA.py`:

- **Data Loading**: Supports both CSV and Excel file formats.
- **Data Processing**:
  - Groups data by `city_name` or `party_name`.
  - Aggregates values using sum or other aggregation functions.
  - Removes sparse columns based on a specified threshold.
- **Dimensionality Reduction**: Uses PCA to reduce the dataset to two dimensions.
- **Visualization**: Generates scatter plots using Plotly to visualize the relationships in the reduced dataset.

### How to Run `PCA.py`:

To execute the script, use the following command:

```bash
python PCA.py
```

The script contains example calls, such as:

```python
compare_cities("ex1/knesset_25.xlsx", 20000)
compare_parties("ex1/knesset_25.xlsx", 20000)
```

### Example Workflow:
1. Place your dataset (CSV or Excel) in the appropriate directory.
2. Modify the file path and threshold in the script if necessary.
3. Run the script to process the data and visualize the results.


## UI_PCA.py - Streamlit Application

The `UI_PCA.py` script is a Streamlit web application that allows users to upload a dataset (CSV or Excel), choose different processing options, and apply dimensionality reduction (PCA) to the data. The application visualizes the reduced dataset in an interactive scatter plot, showing the principal components in a two-dimensional space.

### Key Features of `UI_PCA.py`:

- **Data Upload**: Users can upload CSV or Excel files through the Streamlit interface.
- **Data Preview**: Displays a preview of the uploaded dataset.
- **Processing Options**:
  - Users can choose to process the data either **City-wise** or **Party-wise**.
  - The application provides options to group the data by a selected column (e.g., `city_name` or `party_name`), and apply different aggregation functions such as sum, mean, max, or min.
- **Dimensionality Reduction**: Users can select the number of principal components for dimensionality reduction (PCA).
- **Interactive Visualization**: The app generates an interactive scatter plot using Plotly to visualize the reduced dataset in two dimensions, allowing users to explore the relationships between cities or parties.

### Application Workflow:
1. Upload your dataset (CSV or Excel).
2. Select the column to group by (e.g., `city_name` or `party_name`).
3. Choose the aggregation function (sum, mean, max, or min).
4. Select the number of principal components for dimensionality reduction.
5. Choose the processing type (City-wise or Party-wise).
6. Set the sparsity threshold for removing sparse columns.
7. Click on **Process Data** to see the reduced dataset and the interactive visualization.

## Requirements

Before running the project, ensure that you have the following dependencies installed:

- Python 3.x
- Pandas
- NumPy
- Plotly.express
- Streamlit

You can install the required packages using `pip`:

```bash
pip install pandas numpy plotly streamlit
```

## How to Run the Streamlit Application:

To launch the `UI_PCA.py` application, use the following command:

```bash
streamlit run UI_PCA.py
```

