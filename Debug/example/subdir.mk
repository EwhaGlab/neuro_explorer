################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../example/neuro_explorer_node.cpp \
../example/scan_to_ptcloud_node.cpp \
../example/viz_helper_node.cpp 

CPP_DEPS += \
./example/neuro_explorer_node.d \
./example/scan_to_ptcloud_node.d \
./example/viz_helper_node.d 

OBJS += \
./example/neuro_explorer_node.o \
./example/scan_to_ptcloud_node.o \
./example/viz_helper_node.o 


# Each subdirectory must supply rules for building sources it contributes
example/%.o: ../example/%.cpp example/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/opt/ros/noetic/include -I/home/hankm/catkin_ws/install/include -I/usr/include/opencv4 -include/opt/ros/noetic/include -include/home/hankm/catkin_ws/install/include -include/usr/include/opencv4 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-example

clean-example:
	-$(RM) ./example/neuro_explorer_node.d ./example/neuro_explorer_node.o ./example/scan_to_ptcloud_node.d ./example/scan_to_ptcloud_node.o ./example/viz_helper_node.d ./example/viz_helper_node.o

.PHONY: clean-example

