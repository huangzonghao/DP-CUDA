.PHONY : all clean

all:
	nvcc -v -D__STRICT_ANSI__ \
		src/dynamic_programming.cpp \
		src/dynamic_programming.cu -o dynamic_programming

clean:
	rm dynamic_programming
