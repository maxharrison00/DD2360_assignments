# Report
Report is written up in the `REPORT.md` file.

To compile the report using required naming convention:
```
pandoc -o DD2360HT25_HW1_Group3.pdf REPORT.md
```

# Code

Run `make` to compile the two CUDA source files in Q2 and Q3. This will compile the files using NVCC with the `-arch=sm_75` architecture flag.
These executables can then be run using the executable name with any input, e.g. `./vecAdd 64`.
Profiling is done with `ncu` and `nvprof`.
Logging is done with Bash scripts that can be run with `bash Q2/logging.sh`.

