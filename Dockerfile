FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install necessary packages
RUN apt-get update && \
    apt-get -y install gcc mono-mcs cron && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt ./

#RUN ls -la ./

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app
RUN echo "Contents of /app:" && ls -la /app

# Add the app directory to PYTHONPATH
ENV PYTHONPATH="/app:${PYTHONPATH:-}"

# Command to run the application
CMD ["streamlit", "run", "/app/main.py"]

