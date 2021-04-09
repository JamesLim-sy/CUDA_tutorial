#!/bin/bash
rm  test_load_store  test_script
nvcc   test_load_store.cu  -arch=sm_70   -o test_load_store  2>&1 | grep error
nvcc  -o  test_script  ./test_scripts.cu