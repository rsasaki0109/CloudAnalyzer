FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dri libglu1-mesa libegl1 libxrandr2 libxss1 \
    libxcursor1 libxcomposite1 libxi6 libxtst6 xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY cloudanalyzer/ .

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["ca"]
