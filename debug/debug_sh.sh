#!/bin/bash
rm  debug_op
nvcc  -o  debug_op  -arch=sm_70   ./debug.cu  -lcublas  2>&1 | grep error
