```bash
#!/bin/bash

echo "===================================="
echo "AgriSense AI - Smart Farming Agent"
echo "===================================="

echo "Installing dependencies..."

pip install -q langchain
pip install -q langchain-community
pip install -q chromadb
pip install -q sentence-transformers
pip install -q pypdf
pip install -q gradio

echo "Dependencies installed successfully."

echo "Starting AgriSense AI..."

python main.py
```
