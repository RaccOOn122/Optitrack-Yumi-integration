import pandas as pd

def preprocess_data(input_file, output_file, reference_rigid_body_name):
    """
    Preprocess gesture data by computing relative positions and retaining only necessary columns.

    Args:
        input_file (str): Path to the input CSV file with global marker positions.
        output_file (str): Path to save the processed CSV file.
        reference_rigid_body_name (str): The name of the reference rigid body.
    """
    # Load data
    df = pd.read_csv(input_file, sep=';')

    # Extract reference marker position
    ref_marker = df[df['Rigid Body Name'] == reference_rigid_body_name][['Position X', 'Position Y', 'Position Z']].values
    if ref_marker.size == 0:
        raise ValueError(f"Reference rigid body '{reference_rigid_body_name}' not found in data.")
    ref_marker = ref_marker[0]  # Assuming one reference point

    # Compute relative positions
    df['Relative X'] = df['Position X'] - ref_marker[0]
    df['Relative Y'] = df['Position Y'] - ref_marker[1]
    df['Relative Z'] = df['Position Z'] - ref_marker[2]

    # Round relative values to 4 decimal places
    df['Relative X'] = df['Relative X'].round(4)
    df['Relative Y'] = df['Relative Y'].round(4)
    df['Relative Z'] = df['Relative Z'].round(4)

    # Keep only necessary columns
    df = df[['Frame Number', 'Timestamp', 'Rigid Body Name', 'Relative X', 'Relative Y', 'Relative Z']]

    # Save the processed data
    df.to_csv(output_file, index=False, sep=';')
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    # Process multiple files in a loop
    # for i in range(1, 21):
    #     input_file = f"SortedClap/sorted_clap_{i:02d}.csv"
    #     output_file = f"ProcessedSortedClap/processed_sorted_clap_{i:02d}.csv"
    #     preprocess_data(input_file, output_file, reference_rigid_body_name="Head")

    for i in range(1, 21):
        input_file = f"SortedWave/sorted_wave_{i:02d}.csv"
        output_file = f"ProcessedSortedWave/processed_sorted_wave_{i:02d}.csv"
        preprocess_data(input_file, output_file, reference_rigid_body_name="Head")   
