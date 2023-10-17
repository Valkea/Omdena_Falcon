FROM python:3.10.12-slim

ENV PORT=8000
EXPOSE 8000

# --- start to install backend-end stuff
RUN mkdir -p /app
WORKDIR /app

# --- Install Python requirements.
COPY requirements-frontend.txt ./
RUN pip install --no-cache-dir -r requirements-frontend.txt

# --- Copy project files
COPY ["app.py", "./"]
CMD ["python", "app.py"]

# ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]
# CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout=60 --threads=2
