#!/bin/bash
hipcc -ggdb -I /usr/include/opencv4 `pkg-config opencv4 --cflags --libs` -L /opt/rocm/miopen/lib/ -lMIOpen grayscale.cpp -o grayscale
