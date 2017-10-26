CXX = $(shell root-config --cxx)
LD = $(shell root-config --ld)
INC = $(shell pwd)
Repo = $(shell git rev-parse --show-toplevel)
SRC = $(Repo)/src

CPPFLAGS := $(shell root-config --cflags) -I$(INC)/include 
LDFLAGS := $(shell root-config --glibs) 
CPPFLAGS += -g -std=c++14

TARGET = wwz_ana
TARGET1 = copyTree
TARGET2 = CreateDataCard
TARGET3 = MakePlots

SRC = app/main_WWZAnalysis.cc src/WWZAnalysis.cc
SRC1 = app/copyTree.cc
SRC2 = app/CreateDataCard.cc src/WWZAnalysis.cc
SRC3 = app/MakePlots.cc

OBJ = $(SRC:.cc=.o)
OBJ1 =$(SRC1:.cc=.o)
OBJ2 =$(SRC2:.cc=.o)
OBJ3 =$(SRC3:.cc=.o)

all : $(TARGET) $(TARGET1) $(TARGET2) $(TARGET3)

$(TARGET) : $(OBJ)
	@echo $@
	$(LD) $(CPPFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)

$(TARGET1) : $(OBJ1)
	@echo $@
	$(LD) $(CPPFLAGS) -o $(TARGET1) $(OBJ1) $(LDFLAGS)

$(TARGET2) : $(OBJ2)
	@echo $@
	$(LD) $(CPPFLAGS) -o $(TARGET2) $(OBJ2) $(LDFLAGS)

$(TARGET3) : $(OBJ3)
	@echo $@
	$(LD) $(CPPFLAGS) -o $(TARGET3) $(OBJ3) $(LDFLAGS)

%.o : %.cc
	@echo $@
	$(CXX) $(CPPFLAGS) -o $@ -c $<
clean :
	rm -f *.o src/*.o $(Aux)/src/*.o $(TARGET) $(TARGET1) $(TARGET2) $(TARGET3) *~
