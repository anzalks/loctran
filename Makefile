PYTHON ?= python

.PHONY: screenshots demo

screenshots:
	$(PYTHON) scripts/capture_screenshots.py

demo:
	$(PYTHON) scripts/make_demo_gif.py
