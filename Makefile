#program name
PROGRAM=main

#build file
OBJDIR = build

#c++ compiler
CC = g++
CFLAGS = -I . -I /home/speky/opencv/opencv-3.4.10/include -I /usr/local/cuda/include -g3 -Wall -c -fmessage-length=0 -std=c++11
CSOURCES := $(wildcard *.cpp)
COBJECTS := $(CSOURCES:%.cpp=$(OBJDIR)/%.o)

#CUDA compiler
NVCC = /usr/local/cuda-10.2/bin/nvcc
CUDAPATH = /usr/local/cuda-10.2
NFLAGS = -ccbin g++ -std=c++11 -m64 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_62,code=compute_62
NSOURCES := $(wildcard *.cu)
NOBJECTS := $(NSOURCES:%.cu=$(OBJDIR)/%.o)

#linker
LINKER = g++
LFLAGS = -L/home/speky/opencv/opencv-3.4.10/build/lib -L$(CUDAPATH)/lib64 -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_cudafilters -lopencv_cudaimgproc -lopencv_cudafilters -lcudart -lopencv_features2d -lopencv_cudaarithm -lopencv_cudalegacy -std=c++11 -lpthread 

# colors
Color_Off='\033[0m'
Black='\033[1;30m'
Red='\033[1;31m'
Green='\033[1;32m'
Yellow='\033[1;33m'
Blue='\033[1;34m'
Purple='\033[1;35m'
Cyan='\033[1;36m'
White='\033[1;37m'


all: $(PROGRAM)

$(PROGRAM): $(COBJECTS) $(NOBJECTS)
	@$(LINKER) $(COBJECTS) $(NOBJECTS) -o $@ $(LFLAGS)
	@echo $(Yellow)"Linking complete!"$(Color_Off)

$(COBJECTS): $(OBJDIR)/%.o : %.cpp
	@echo $(Blue)"C++ compiling "$(Purple)$<$(Color_Off)
	@$(CC) $(CFLAGS) -c $< -o $@
	@echo $(Blue)"C++ compiled "$(Purple)$<$(Blue)" successfully!"$(Color_Off)
	
$(NOBJECTS): $(OBJDIR)/%.o : %.cu
	@echo $(Green)"CUDA compiling "$(Purple)$<$(Color_Off)
	@$(NVCC) $(NFLAGS) -c $< -o $@
	@echo $(Green)"CUDA compiled "$(Purple)$<$(Green)" successfully!"$(Color_Off)

clean:
	@rm -f $(PROGRAM) $(COBJECTS) $(NOBJECTS)
	@echo $(Cyan)"Cleaning Complete!"$(Color_Off)
