
HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
        HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc


CXX := $(HIPCC)
TARGET := conv
ROCM_PATH := /opt/rocm
HEADERS := -I $(ROCM_PATH)/include -I /usr/local/include/opencv4
LIBS := -L $(ROCM_PATH)/lib -L/usr/local/lib
CXXFLAGS := -std=c++17 -O2

all: conv

conv: $(TARGET).cpp
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(TARGET).cpp -o $(TARGET) \
            -lMIOpen -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

.phony: clean

clean:
	rm $(TARGET) || echo -n ""



