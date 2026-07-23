# Loctran — Private AI PDF Translator
# Copyright (C) 2026 Anzal K Shahul
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Loctran package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("loctran")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0.dev0"
