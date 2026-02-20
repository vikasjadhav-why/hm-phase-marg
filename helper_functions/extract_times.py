import re
import sys
import h5py
import numpy as np
import glob

pattern = sys.argv[1]  # e.g. "out_*.out"
files = sorted(glob.glob(pattern))

all_numbers = []

for filename in files:
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if re.match(r'^-?\d+\.?\d*([eE][+-]?\d+)?$', line):
                all_numbers.append(float(line))
    print(f"Processed {filename}")

with h5py.File('time.hdf', 'w') as hf:
    hf.create_dataset('times', data=np.array(all_numbers))

print(f"Stored {len(all_numbers)} numbers total in time.hdf")
