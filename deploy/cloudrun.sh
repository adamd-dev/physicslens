#!/usr/bin/env bash
# Deploy PhysicsLens as a public Cloud Run service with History and videos kept in a Cloud Storage bucket.
# Usage: deploy/cloudrun.sh <project-id>   (after `gcloud init` and linking billing)
# MODEL_API_KEY and PHYSICSLENS_MAX_USD are set by you in the Cloud Run console, never here.
set -euo pipefail

PROJECT="${1:?usage: deploy/cloudrun.sh <project-id>}"
REGION="us-central1"                 # the Cloud Storage free tier is in us-central1, us-east1 and us-west1
SERVICE="physicslens"
BUCKET="${PROJECT}-physicslens-data"

cd "$(dirname "$0")/.."
gcloud config set project "$PROJECT"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com storage.googleapis.com

# History, uploaded videos, model caches and the spend log live in the bucket: they survive the app sleeping.
gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1 \
  || gcloud storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access
NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="serviceAccount:${NUMBER}-compute@developer.gserviceaccount.com" --role=roles/storage.objectAdmin >/dev/null

# One instance: History is one shared list and one analysis runs at a time. Scales to zero when idle.
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 7860 \
  --cpu 2 --memory 4Gi \
  --timeout 900 \
  --min-instances 0 --max-instances 1 --concurrency 20 \
  --execution-environment gen2 \
  --add-volume "name=data,type=cloud-storage,bucket=$BUCKET,mount-options=uid=1000;gid=1000" \
  --add-volume-mount "volume=data,mount-path=/app/cache" \
  --update-env-vars "PHYSICSLENS_MAX_UPLOADS=20"

gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'
