FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy the rest of the application
COPY . .

# Add SSH key to the container
COPY ~/.ssh /root/.ssh

# Set correct permissions for the SSH key
RUN chmod 600 /root/.ssh/*

#Copy Global git config
COPY ~/.gitconfig /root/.gitconfig

# Command to run the application
CMD ["python3", "ResNet/main_pruning.py"]

