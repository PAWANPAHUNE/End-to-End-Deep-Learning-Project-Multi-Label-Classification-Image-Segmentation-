# Use a lightweight Python base image
FROM python:3.10-slim

# Install system dependencies (add awscli back here if you specifically need to download data from S3)
RUN apt-get update -y && apt-get install -y gcc

# Set up a non-root user (Required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user

# Set environment variables for the user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy the requirements file and install dependencies
# We do this before copying the rest of the app to leverage Docker caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application files with user permissions
COPY --chown=user . $HOME/app

# Hugging Face natively exposes port 7860
# If your app.py is configured to use Gradio's default port, this will work perfectly.
CMD ["python", "app.py"]