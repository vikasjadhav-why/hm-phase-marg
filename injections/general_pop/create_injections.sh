#!/bin/bash


# Usage: ./run_injections.sh <N> <config-file>
# Runs pycbc_create_injections N times, with seed and output file named by iteration index.

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <N> <config-file>"
    echo "  N            Number of injections to generate (iterations 0 to N-1)"
    echo "  config-file  Path to the .ini config file"
    exit 1
fi

N=$1
CONFIG=$2


echo "Running pycbc_create_injections for n = 0 to $((N - 1))..."

for (( n=0; n<N; n++ )); do
    echo "--- Iteration n=$n ---" 
    pycbc_create_injections \
        --config-files "$CONFIG" \
        --ninjections 1 \
        --seed "$n" \
        --output-file "injection_${n}.hdf" \
        --variable-params-section variable_params \
        --static-params-section static_params \
        --force

    if [ $? -ne 0 ]; then
        echo "Error: pycbc_create_injections failed at n=$n. Aborting."
        exit 1
    fi
done

echo "Done. Generated $N injection files: injection_0.hdf ... injection_$((N-1)).hdf"