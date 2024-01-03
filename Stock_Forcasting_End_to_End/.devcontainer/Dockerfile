# Use the official Python image from the Codespaces container registry
FROM ghcr.io/codespaces-images/vscode-remote-python

# Install fbprophet dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev

# Install fbprophet
RUN pip install fbprophet
