all:
	cython -a *.pyx
	python setup.py build_ext --inplace
	python ./test_dp_state.py

clean:
	python setup.py clean
	rm -f *.so *.c *.html
