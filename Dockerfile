FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install necessary packages and SQLite3
# Install necessary packages
RUN apt-get update && \
    apt-get -y install gcc mono-mcs cron && \
    rm -rf /var/lib/apt/lists/*


COPY requirements.txt ./

# Upgrade pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Add the app directory to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "main.py"]