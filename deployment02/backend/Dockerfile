FROM python:3.10.12-slim

ENV PORT=5000
EXPOSE 5000

# --- start to install backend-end stuff
RUN mkdir -p /app
WORKDIR /app

# --- Install Python requirements.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy project files
COPY ["run.py", "./"]

# --- Start server
# ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:$PORT", "run:app"]
CMD gunicorn run:app --bind 0.0.0.0:$PORT --timeout=60 --threads=2
