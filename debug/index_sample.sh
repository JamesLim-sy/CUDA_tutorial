#!/bin/bash
rm  index_op
nvcc  -o  index_op  -arch=sm_70   ./index_sample.cu  -lcublas  2>&1 | grep error
