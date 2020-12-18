#!/bin/bash
rm element_add
nvcc  -o  element_add  ./src/*.cu  -lcublas  2>&1 | grep error
