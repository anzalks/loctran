PYTHON ?= python

.PHONY: screenshots
screenshots:
	$(PYTHON) scripts/capture_screenshots.py
