WHL_BUILD_DIR := package

# default rule
default: whl

.PHONY: whl
whl:
	python setup.py sdist bdist_wheel

.PHONY: clean
clean:
	rm -rf $(WHL_BUILD_DIR) build dist *.egg-info
