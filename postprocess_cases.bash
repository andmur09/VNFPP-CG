#!/bin/bash
path=$(pwd)
results=$path/data_used/results
input=$path/to_process.txt

ulimit -t 1800
ulimit -v 70000000

while IFS= read -r file; do
    echo "Processing Problem ${file}";
    python  ${path}/postprocess.py ${results}/${file} ${path}/results.csv;
done < "$input"