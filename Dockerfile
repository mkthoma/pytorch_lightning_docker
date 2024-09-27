# Use NVIDIA CUDA image as the base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Set the PYTHONPATH to include the src directory
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Create a directory for Kaggle credentials
RUN mkdir -p /root/.kaggle

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
# Set up Kaggle credentials\n\
echo "{\\"username\\":\\"$KAGGLE_USERNAME\\",\\"key\\":\\"$KAGGLE_KEY\\"}" > /root/.kaggle/kaggle.json\n\
chmod 600 /root/.kaggle/kaggle.json\n\
\n\
if [ "$1" = "train" ]; then\n\
    python3 src/train.py "${@:2}"\n\
elif [ "$1" = "eval" ]; then\n\
    python3 src/eval.py "${@:2}"\n\
elif [ "$1" = "infer" ]; then\n\
    python3 src/infer.py "${@:2}"\n\
else\n\
    echo "Invalid command. Use 'train', 'eval', or 'infer'."\n\
    exit 1\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]