.PHONY: all setup download generation clean

all: setup download generation

setup:
	pipenv install

download:
	python import_dataset.py

generation:
	python generate_images.py

clean:
	rm -rf images
