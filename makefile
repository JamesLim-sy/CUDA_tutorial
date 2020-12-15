#USER SPECIFIC DIRECTORIES ##
# CUDA directory:
CUDA_ROOT_DIR=/home/baidu/personal-code/CUDA_homework/
##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS=
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc

# CUDA library directory:
CUDA_LIB_DIR= -L $(CUDA_ROOT_DIR)/lib
# CUDA include directory:
CUDA_INC_DIR=  $(CUDA_ROOT_DIR)/inc
CUDA_BIN_DIR=  $(CUDA_ROOT_DIR)/bin

# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

NVCC_FLAGS= -I$(CUDA_INC_DIR)
NVCC_LIBS=

##########################################################

## Project file structure ##
# Source file directory:
CUDA_FILE = $(wildcard *.cu)

# Object file directory:
OBJ_FILE = $(wildcard *.o)

##########################################################

## Make variables ##
# Target executable name:
PRJ_NAME = element_wise

# Object files:
OBJ_PATH = $(CUDA_OBJ_DIR)/*.o 

##########################################################

## Compile ##
all : build

build : gpu
	$(NVCC)  $(NVCC_FLAGS) -o $(OBJ_FILE)

gpu:
	$(NVCC)  $(NVCC_FLAGS)  $(CUDA_FILE)

# Clean objects in object directory.
clean:
	rm $(OBJ_FILE)  $(PRJ_NAME)

