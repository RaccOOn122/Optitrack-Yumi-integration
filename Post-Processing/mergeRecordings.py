import pandas as pd
import glob

def merge_and_label(input_dirs, labels, output_file):
    """
    Merge multiple gesture data files from specified directories and add gesture labels.

    Args:
        input_dirs (list): List of directories containing gesture files.
        labels (list): Corresponding labels for each directory.
        output_file (str): Path to save the merged file.
    """
    if len(input_dirs) != len(labels):
        raise ValueError("Each directory must have a corresponding label.")

    merged_data = []

    for directory, label in zip(input_dirs, labels):
        # Get all CSV files from the directory
        files = glob.glob(f"{directory}/processed_sorted_*.csv")
        
        for file in files:
            # Load the data
            df = pd.read_csv(file, sep=';')

            # Add the gesture label
            df['Gesture'] = label

            # Append to the merged data list
            merged_data.append(df)

    # Combine all dataframes
    merged_df = pd.concat(merged_data, ignore_index=True)

    # Save the merged file
    merged_df.to_csv(output_file, index=False, sep=';')
    print(f"Merged data saved to {output_file}")

if __name__ == "__main__":
    # Directories and labels
    input_dirs = ["ProcessedSortedClap", "ProcessedSortedWave"]
    labels = ["clap", "wave"]

    # Output file
    output_file = "merged_gesture_data.csv"

    # Merge and label
    merge_and_label(input_dirs, labels, output_file)
