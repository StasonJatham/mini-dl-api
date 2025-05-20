# Meow

## Start 

```bash
python -m .venv venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
uvicorn main:app
spotdl web --host 127.0.0.1 --port 8800 --keep-alive
```