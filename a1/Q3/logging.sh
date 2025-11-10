#!/bin/bash

mkdir -p logs

# Collect performance data for various sizes
for expon in 10 11 12 13 14; do
  let arr_len=$((2**expon))
  echo "Array length: ${arr_len}"
  nvprof --log-file logs/nvprof_${expon}.log --csv --normalized-time-unit us --print-gpu-trace ./matrixMultiplication $arr_len $arr_len $arr_len
done

# Parse the log files to extract timings
echo "ArrayLengthLog,HtoD_Time,Kernel_Time,DtoH_Time" > results.csv
for expon in 10 11 12 13 14; do
    h2d=$(grep "HtoD" logs/nvprof_${expon}.log | awk -F',' '{sum += $2} END {print sum}')
    kernel=$(grep "multiplication" logs/nvprof_${expon}.log | awk -F',' '{print $2}')
    d2h=$(grep "DtoH" logs/nvprof_${expon}.log | awk -F',' '{print $2}')
    echo "$expon,$h2d,$kernel,$d2h"
    echo "$expon,$h2d,$kernel,$d2h" >> results.csv
done

echo "Results saved to results.csv"
