#USER SPECIFIC DIRECTORIES ##
# CUDA directory:
CUDA_ROOT_DIR=/home/baidu/personal-code/CUDA_homework
##########################################################
## NVCC COMPILER OPTIONS ##
NVCC:=nvcc
TARGET=element_wise

# CUDA library directory:
CUDA_SRC_DIR=  $(CUDA_ROOT_DIR)/src
CUDA_INC_DIR=  $(CUDA_ROOT_DIR)/inc

NVCCFLAGS =  -I$(CUDA_INC_DIR)  -arch=sm_70
LDFLAGS=  -lcublas 

##########################################################
CUDA_FILE = $(wildcard  $(CUDA_SRC_DIR)/*.cu )
OBJ = $(patsubst %.cu, %.o, $(CUDA_FILE) )

##########################################################
## Compile ##
.PHONY: gpu clean

gpu :$(TARGET)

$(TARGET) : $(OBJ)
	$(NVCC) -o $@  $^   $(LDFLAGS)   -arch=sm_70
	$ echo ">> make success!\n"

%.o:%.cu
	$(NVCC) -c $<  -o $@  $(NVCCFLAGS) 

clean: 
	rm  $(TARGET)  $(OBJ)