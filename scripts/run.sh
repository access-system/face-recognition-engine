#!/usr/bin/env bash

docker-compose -f docker/docker-compose.yml up --build -d

if [ -d ".venv/" ]; then
    source .venv/bin/activate
else
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

python3 cmd/main.py