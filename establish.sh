#!/bin/bash
rm element_add
nvcc  -o  element_add  ./src/*.cu   2>&1 | grep error
