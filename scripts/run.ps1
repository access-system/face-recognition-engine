docker-compose -f "docker\docker-compose.yaml" up --build -d

$ENV:PYTHONPATH = $PWD

if (Test-Path -Path ".\.venv" -PathType Container) {
    .\.venv\Scripts\Activate.ps1
} else {
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
}

python "cmd\main.py"