#!/bin/bash

# Stop on any error
set -e

# --- Configuration ---
# Make sure this path is correct for your VM's file structure
PROJECT_DIR="/home/roman/yes-partener"

# --- Deployment ---

echo "Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# Activate the virtual environment
# This is the crucial step to ensure you're using the correct Python environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install or update dependencies
# It's a good practice to ensure all dependencies are installed.
# If you don't have a requirements.txt file, you can create one with 'pip freeze > requirements.txt'
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Cleaning media directory..."
rm -rf media/*

echo "Removing database..."
rm -f db.sqlite3

echo "Making migrations..."
python3 manage.py makemissions

echo "Applying migrations..."
python3 manage.py migrate

echo "Starting the Django server..."
# This will run the development server. For a real-world application,
# you would use a production-ready server like Gunicorn or uWSGI.
python3 manage.py runserver 0.0.0.0:8000