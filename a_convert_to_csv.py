from pathlib import Path
import numpy as np
import pandas as pd
import wfdb  # WaveForm-Database package


def convert_to_csv(path='./', new_folder=True):
    """
    Code to convert all .dat files (ECG signals) in a folder to CSV format
    Path - location of the dat files (str)
    new_folder - whether to put the files in a new csv folder (boolean)
    """
    print('Reading files...')
    if new_folder:
        Path(path).joinpath("csv_files").mkdir(parents=True, exist_ok=True)
    # Get list of all .dat files in the current folder
    for file in read_data(path=path, extension="*.dat"):
        print(f"Processing file: {file}")
        filename = file.stem
        # Read the signal data
        record, fields = wfdb.rdsamp(str(file.with_suffix("")))  # Use base name without .dat
        # Fix misspelled 'hand GSr' in the signal names
        fields['sig_name'] = ["hand GSR" if s == "hand GSr" else s for s in fields['sig_name']]
        # Column headers
        header = ",".join(["-".join(x) for x in zip(fields['sig_name'], fields['units'])])
        # Add to the corresponding folder
        output_file = Path(path).joinpath("csv_files") / f"{filename}.csv" if new_folder else f"{filename}.csv"
        np.savetxt(output_file, record, delimiter=",", header=header, comments="")
    print('All files read successfully!')


def read_data(path, extension="*.csv"):
    """
    Function to read the all the csv/dat files.
    Since there are many files, it returns an iterator.
    path - path to the csv/dat files.
    extension - csv/dat. (if csv, it returns the dataframe else only filename)
    """
    file_paths = Path(path).glob(extension)
    for file in file_paths:
        if extension == "*.csv":
            dataset = pd.read_csv(file)
            yield file.stem, dataset
        else:
            yield file


if __name__ == "__main__":
    convert_to_csv("D:/MLdata/stress-recognition-in-automobile-drivers-1.0.0")
