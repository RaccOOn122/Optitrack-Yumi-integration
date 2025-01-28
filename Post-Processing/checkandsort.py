import pandas as pd

def check_and_sort_frame_order(input_file, output_file):
    """
    Check if the 'Frame Number' and 'Timestamp' columns are in order in the given CSV file.
    If not, sort the rows based on 'Frame Number'. Additionally, ensure that each frame number has a maximum
    of 3 rows with different rigid body names (Head, HandRight, HandLeft).

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the sorted and cleaned CSV file.

    Returns:
        None: Prints the status and saves the sorted file if necessary.
    """
    # Load the CSV file in chunks to handle large files
    chunks = pd.read_csv(input_file, sep=';', chunksize=100000)

    # Combine all chunks into a single DataFrame
    df = pd.concat(chunks, ignore_index=True)

    # Ensure the DataFrame has necessary columns
    if 'Frame Number' not in df.columns or 'Rigid Body Name' not in df.columns:
        print("Error: Required columns 'Frame Number' or 'Rigid Body Name' are missing.")
        return

    # Check if 'Frame Number' is sorted
    frame_sorted = df['Frame Number'].is_monotonic_increasing

    if not frame_sorted:
        print(f"Frame numbers are not in order in {input_file}. Sorting the file.")
        # Sort the DataFrame by 'Frame Number' and 'Rigid Body Name'
        df.sort_values(by=['Frame Number', 'Rigid Body Name'], inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Limit to max 3 rows per frame with different rigid body names
    def limit_to_unique_rigid_bodies(group):
        unique_bodies = group.drop_duplicates(subset='Rigid Body Name')
        return unique_bodies.head(3)

    df = df.groupby('Frame Number', group_keys=False).apply(limit_to_unique_rigid_bodies).reset_index(drop=True)

    # Save the sorted and cleaned DataFrame to a new file
    df.to_csv(output_file, index=False, sep=';')
    print(f"Sorted and validated data saved to {output_file}")

if __name__ == "__main__":
    # Process multiple files in a loop
    # for i in range(1, 21):
    #     input_file = f"RawClap/clap_{i:02d}.csv"
    #     output_file = f"SortedClap/sorted_clap_{i:02d}.csv"
    #     check_and_sort_frame_order(input_file, output_file)

    for i in range(1, 21):
        input_file = f"RawWave/wave_{i:02d}.csv"
        output_file = f"SortedWave/sorted_wave_{i:02d}.csv"
        check_and_sort_frame_order(input_file, output_file)
