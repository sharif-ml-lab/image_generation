import os
import pandas as pd


def clean_and_sort_column(column):
    """
    Cleans the newline characters from each item in the column,
    then sorts the column based on the length of the items.
    """
    cleaned = [item.replace('\n', ' ') for item in column]
    cleaned.sort(key=lambda item: len(item))
    return cleaned


# Directory containing the CSV files
dir_name = 'GPT'

# Check if the directory exists
if not os.path.exists(dir_name):
    print(f"The directory {dir_name} does not exist.")
else:
    # Iterate over all CSV files in the directory
    for filename in os.listdir(dir_name):
        if filename.endswith('.csv'):
            # Construct the full path to the file
            file_path = os.path.join(dir_name, filename)
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Extract the base name (without extension) for naming new files
            base_name = filename.rsplit('.', 1)[0]
            
            # Iterate over the columns and create a new CSV file for each
            for column in df.columns:
                if column not in ['exp_sgl']:
                    continue
                # Create a new DataFrame for the current column
                cleaned_column = clean_and_sort_column(df[column])
                
                # Create a new DataFrame for the cleaned and sorted column
                new_df = pd.DataFrame(cleaned_column[3:18], columns=['caption'])
                
                # Construct the name for the new CSV file
                new_filename = f"{base_name}_{column}.csv"
                new_file_path = os.path.join('GPT_IND', new_filename)
                
                # Save the DataFrame to a new CSV file
                new_df.to_csv(new_file_path, index=False)
                print(f"Created file: {new_file_path}", len(new_df['caption']))
