FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    libgl1-mesa-dri libglu1-mesa libegl1 libxrandr2 libxss1 \
    libxcursor1 libxcomposite1 libxi6 libxtst6 xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

ARG CLOUDANALYZER_REF=main
RUN pip install --no-cache-dir \
    "git+https://github.com/rsasaki0109/CloudAnalyzer.git@${CLOUDANALYZER_REF}#subdirectory=cloudanalyzer"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
