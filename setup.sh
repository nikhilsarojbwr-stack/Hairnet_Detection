#!/bin/bash
# Uninstall any existing opencv packages (including cached ones)
uv pip uninstall -y opencv-python opencv-python-headless
# Install the headless version fresh
uv pip install --no-cache-dir opencv-python-headless==4.9.0.80
