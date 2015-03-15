.PHONY : all clean

all:
	nvcc -v -ccbin /usr/bin/clang -D__STRICT_ANSI__ \
		src/dynamic_programming.cpp \
		src/dynamic_programming.cu -o dynamic_programming

clean:
	rm dynamic_programming
