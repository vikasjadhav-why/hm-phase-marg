import h5py
import argparse
import glob
import os

def combine_hdf_files(input_files, output_file):
    """
    Combine multiple output HDF files into a single file, maintaining the same structure.
    Inputs:
    "input_files" : list of input HDF files to combine
    "output_file" : path to the output combined HDF file
    """
    with h5py.File(output_file, 'w') as out_f:
        attr_set = False
        for input_file in input_files:
            print(f"Processing {input_file}...")
            with h5py.File(input_file, 'r') as in_f:
                # Set the root attribute only once from the first file
                if not attr_set:
                    out_f.attrs['injection_folder'] = in_f.attrs['injection_folder']
                    attr_set = True

                # Copy each injection group to the output file
                for group_name in in_f.keys():
                    if group_name in out_f:
                        print(f"  Warning: {group_name} already exists in output, skipping...")
                        continue
                    in_f.copy(group_name, out_f)

    print(f"Done. Combined file written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Combine multiple output HDF files into one.")
    parser.add_argument("output_file", help="path to the combined output HDF file")
    parser.add_argument("input_files", nargs='+', help="input HDF files to combine, e.g. output_*.hdf")

    args = parser.parse_args()

    # Expand any glob patterns (e.g. output_*.hdf) in case shell doesn't auto-expand
    input_files = []
    for pattern in args.input_files:
        matched = sorted(glob.glob(pattern))
        if not matched:
            print(f"Warning: no files matched pattern '{pattern}'")
        input_files.extend(matched)

    if not input_files:
        print("Error: no input files found")
        return

    print(f"Found {len(input_files)} files to combine:")
    for f in input_files:
        print(f"  {f}")

    combine_hdf_files(input_files, args.output_file)


if __name__ == "__main__":
    main()
