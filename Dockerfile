FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install necessary packages and SQLite3
RUN apt-get update && \
    apt-get -y install gcc mono-mcs cron wget build-essential libsqlite3-dev && \
    wget https://www.sqlite.org/2023/sqlite-autoconf-3420000.tar.gz && \
    tar xvfz sqlite-autoconf-3420000.tar.gz && \
    cd sqlite-autoconf-3420000 && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm -rf sqlite-autoconf-3420000 && \
    rm sqlite-autoconf-3420000.tar.gz && \
    ldconfig && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add the app directory to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "main.py"]