# PhysicsLens as a public Hugging Face Space: the web app built with Node, served by the Python API on port 7860.
FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.13-slim
# OpenCV's headless wheel still needs GLib at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user pyproject.toml cli.py ./
COPY --chown=user physicslens/ physicslens/
# Editable: the package finds cache/, costs.csv and web/dist under /app (PROJECT_ROOT is the source tree).
RUN pip install --no-cache-dir -e ".[ui]"
COPY --chown=user --from=web /web/dist web/dist
# The spend log lives with the other data in cache/: on Cloud Run that is a bucket, so the budget cap survives restarts.
RUN mkdir -p cache && ln -s cache/costs.csv costs.csv && chown -R user /app
USER user
ENV PHYSICSLENS_HOST=0.0.0.0 \
    PHYSICSLENS_PORT=7860 \
    PHYSICSLENS_PUBLIC=1 \
    PHYSICSLENS_MAX_UPLOAD_MB=100 \
    PHYSICSLENS_MAX_UPLOADS=40
EXPOSE 7860
CMD ["physicslens-ui"]
