#!/bin/bash

# Stop on any error
set -e

# --- Configuration ---
# Make sure this path is correct for your VM's file structure
PROJECT_DIR="/home/roman/yes-partener"

# --- Deployment ---

echo "Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# It's good practice to use a virtual environment
# If you have one, activate it like this:
# source venv/bin/activate

# Also, ensure your dependencies are installed
# pip install -r requirements.txt

echo "Cleaning media directory..."
rm -rf media/*

echo "Removing database..."
rm -f db.sqlite3

echo "Making migrations..."
python3 manage.py makemigrations

echo "Applying migrations..."
python3 manage.py migrate

echo "Starting the Django server..."
# This will run the development server. For a real-world application,
# you would use a production-ready server like Gunicorn or uWSGI.
# The server will stop when you close your terminal.
python3 manage.py runserver 0.0.0.0:8000