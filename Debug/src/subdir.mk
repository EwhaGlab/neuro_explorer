################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/explorer.cpp \
../src/frontier_filter.cpp \
../src/image_data_handler.cpp \
../src/neuro_explorer.cpp \
../src/scan_to_ptcloud.cpp \
../src/viz_helper.cpp 

CPP_DEPS += \
./src/explorer.d \
./src/frontier_filter.d \
./src/image_data_handler.d \
./src/neuro_explorer.d \
./src/scan_to_ptcloud.d \
./src/viz_helper.d 

OBJS += \
./src/explorer.o \
./src/frontier_filter.o \
./src/image_data_handler.o \
./src/neuro_explorer.o \
./src/scan_to_ptcloud.o \
./src/viz_helper.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp src/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/opt/ros/noetic/include -I/home/hankm/catkin_ws/install/include -I/usr/include/opencv4 -include/opt/ros/noetic/include -include/home/hankm/catkin_ws/install/include -include/usr/include/opencv4 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-src

clean-src:
	-$(RM) ./src/explorer.d ./src/explorer.o ./src/frontier_filter.d ./src/frontier_filter.o ./src/image_data_handler.d ./src/image_data_handler.o ./src/neuro_explorer.d ./src/neuro_explorer.o ./src/scan_to_ptcloud.d ./src/scan_to_ptcloud.o ./src/viz_helper.d ./src/viz_helper.o

.PHONY: clean-src

