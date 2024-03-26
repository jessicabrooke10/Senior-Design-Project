# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=web.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run web.py when the container launches
CMD ["flask", "run"]

# Build your Docker image example
# docker build -t my-flask-app .

# Run your Docker container example
# docker run -p 5000:5000 my-flask-app