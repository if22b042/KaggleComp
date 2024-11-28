import pandas as pd
import matplotlib.pyplot as plt

def analyze_correlation(train_data):
    """
    Analyzes the correlation of features in a DataFrame and visualizes it as a heatmap.
    
    Parameters:
        train_data (pd.DataFrame): The input training data with features.
    
    Returns:
        pd.DataFrame: Correlation matrix of the DataFrame.
    """
    # Compute the correlation matrix
    correlation_matrix = train_data.corr()

    # Display the correlation matrix
    print("Correlation Matrix:")
    print(correlation_matrix)

    # Plot the heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="none")
    plt.colorbar(label="Correlation Coefficient")
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    return correlation_matrix


train_data= pd.read_csv('train_set.csv', index_col=0)
matrix=analyze_correlation(train_data=train_data)
