#!/bin/bash

FINAL_BASE="/home/vjadhavy/hmphase/hm-phase-marg/final_results/mode2233"

for top in gen hm; do
    for sub in all dom grid quad weighted; do
        src_out="$top/$sub/out_files/time.hdf"
        src_res="$top/$sub/results/results.hdf"
        dest="$FINAL_BASE/$top/$sub"

        if [ -f "$src_out" ]; then
            cp "$src_out" "$dest/time.hdf"
            echo "Copied $src_out -> $dest/time.hdf"
        else
            echo "WARNING: $src_out not found"
        fi

        if [ -f "$src_res" ]; then
            cp "$src_res" "$dest/results.hdf"
            echo "Copied $src_res -> $dest/results.hdf"
        else
            echo "WARNING: $src_res not found"
        fi
    done
done

echo "Done."
