#!/bin/bash

# Use for personal laptop
bash -c "time seq 16 | parallel ./run_square.sh" 2>&1 | grep real

# Use for HPC
# bash -c "time seq 1024 | parallel ./run_square.sh" 2>&1 | grep real
