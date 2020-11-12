#program name
PROGRAM=main

#build file
OBJDIR = build

#c++ compiler
CC = g++
CFLAGS = -I . -I /home/meli/opencv/opencv-3.4.10/build/include -g3 -Wall -c -fmessage-length=0 -std=c++11
CSOURCES := $(wildcard *.cpp)
COBJECTS := $(CSOURCES:%.cpp=$(OBJDIR)/%.o)

#linker
LINKER = g++
LFLAGS = -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -std=c++11 -lpthread

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
	@$(LINKER) $(COBJECTS) -o $@ $(LFLAGS)
	@echo $(Yellow)"Linking complete!"$(Color_Off)

$(COBJECTS): $(OBJDIR)/%.o : %.cpp
	@echo $(Blue)"C++ compiling "$(Purple)$<$(Color_Off)
	@$(CC) $(CFLAGS) -c $< -o $@
	@echo $(Blue)"C++ compiled "$(Purple)$<$(Blue)" successfully!"$(Color_Off)

clean:
	@rm -f $(PROGRAM) $(COBJECTS) $(NOBJECTS)
	@echo $(Cyan)"Cleaning Complete!"$(Color_Off)