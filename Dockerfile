FROM python:3.12-slim

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin runner
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./
USER runner
CMD ["python", "-u", "/app/app.py"]
