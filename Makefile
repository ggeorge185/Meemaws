.PHONY: all setup run

all: setup run

setup:
	@echo "Setting up the virtual environment and installing dependencies..."
	python3 -m venv venv
	. venv/bin/activate && pip install flask langchain chromadb sentence-transformers langchain-community

run:
	@echo "Starting the Flask server..."
	. venv/bin/activate && FLASK_APP=server.py flask run
