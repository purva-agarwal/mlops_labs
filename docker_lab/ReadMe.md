# Iris Model Training & API (Docker + FastAPI)

This lab demonstrates training a **Random Forest model** on the Iris dataset and serving predictions through a **FastAPI API**, all inside a **Docker container** with a virtual environment.

---

## Lab Structure

```

docker_lab/
│
├── src/
│   ├── __init__.py
│   ├── main.py       # Entry point, trains the model
│   ├── train.py      # Training logic
│   └── api.py        # API for predictions
│
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
├── logs/
├── assets/           # Screenshots

````

---

## Getting Started

### Build and Run the Container

From the project root:

```bash
docker compose up --build
````

This will:

* Create a virtual environment inside the container
* Install dependencies
* Start a FastAPI server on **port 8000**

---

### Access the API

* **API root:** [http://localhost:8000](http://localhost:8000)
* **Interactive docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Test the API Using `curl`

Send a POST request to `/predict`:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Expected response:

```json
{"prediction": 0}
```

> Make sure the container is running and Uvicorn is set to `--host 0.0.0.0` in `CMD` for Docker.

---

### Check Inside the Container

Exec into the container:

```bash
docker exec -it iris-api bash
```

* Check virtual environment Python:

```bash
which python
# Should output: /opt/venv/bin/python
```

* Check installed packages:

```bash
pip list
```

* Optional: Test model training manually:

```bash
python
>>> from src.train import train_model
>>> model = train_model()
```

Exit the container:

```bash
exit
```

---

### Stop the Container

```bash
docker-compose down
```

---

## Tech Stack

* **Python 3.9 (slim)**
* **FastAPI** for API serving
* **Scikit-learn** for ML
* **Docker + Docker Compose** for containerization

---