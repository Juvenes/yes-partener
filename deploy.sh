#!/bin/bash

# Stop on any error
set -e

# --- Configuration ---
PROJECT_DIR="/home/roman/yes-partener"
LOG_FILE="$PROJECT_DIR/django_server.log"
VENV_DIR="$PROJECT_DIR/venv"

# --- Deployment ---

echo "Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# --- Virtual Environment Management ---

# Create a virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  echo "Virtual environment not found. Creating one..."
  python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install -r requirements.txt

# --- Database & Media ---

echo "Cleaning media directory..."
rm -rf media/*

echo "Removing database..."
rm -f db.sqlite3

echo "Making migrations..."
python3 manage.py makemigrations

echo "Applying migrations..."
python3 manage.py migrate

# --- Server Management ---

echo "Stopping any existing Django server..."
pkill -f "manage.py runserver" || echo "No server was running."

echo "Starting the Django server in the background..."
nohup python3 manage.py runserver 0.0.0.0:8000 > "$LOG_FILE" 2>&1 &

echo "Deployment finished. Server is running in the background."
echo "Check logs at: $LOG_FILE"