import pandas as pd

# Function to remove rows with NaN values
def remove_rows_with_nan(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Drop rows with NaN values
    data_cleaned = data.dropna(axis=0, how='any')  # Drop rows with any NaN values
    
    # Save the cleaned data to a new file called fixed_train_set.csv
    cleaned_file_path = "fixed_train_set.csv"
    data_cleaned.to_csv(cleaned_file_path, index=False)
    
    print(f"Cleaned data saved to {cleaned_file_path}")
    return data_cleaned

# Usage
file_path = 'train_set.csv'  # Path to your input CSV file
cleaned_data = remove_rows_with_nan(file_path)

# Show the cleaned data (optional)
print(cleaned_data)
