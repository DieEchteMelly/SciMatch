FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt ./

# Install necessary packages
RUN apt-get update && \
    apt-get -y install gcc mono-mcs cron && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Add the app directory to PYTHONPATH
ENV PYTHONPATH="/app:${PYTHONPATH:-}"

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "main.py"]