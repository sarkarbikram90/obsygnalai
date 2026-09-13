#!/usr/bin/env bash
set -euo pipefail

# Ensure gcloud is in PATH if installed in user local appdata
if ! command -v gcloud &>/dev/null; then
  export PATH="${PATH}:/c/Users/bikrams/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin"
fi

echo "============================================================"
echo " Obsygnal AI: Cloud Run Serverless Compilation & Deployment"
echo "============================================================"

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
SERVICE_NAME="obsygnal"
REGION="${GCP_REGION:-us-central1}"
IMAGE_TAG="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

if [ -z "${PROJECT_ID}" ] || [ "${PROJECT_ID}" = "(unset)" ]; then
  echo "[ERROR] No active GCP project found in gcloud context."
  echo "Please set an active project using: gcloud config set project <PROJECT_ID>"
  exit 1
fi

echo "[INFO] Target Project: ${PROJECT_ID}"
echo "[INFO] Service Name:   ${SERVICE_NAME}"
echo "[INFO] Region:         ${REGION}"
echo "[INFO] Target Image:   ${IMAGE_TAG}"

echo "[STEP 1/2] Triggering remote container compilation via Cloud Build..."
gcloud builds submit --default-buckets-behavior=regional-user-owned-bucket --tag "${IMAGE_TAG}" .

echo "[STEP 2/2] Deploying container to Cloud Run with NVIDIA L4 GPU acceleration..."
gcloud beta run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_TAG}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --cpu 4 \
  --memory 16Gi \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 1 \
  --timeout 600s \
  --no-cpu-throttling \
  --no-gpu-zonal-redundancy \
  --set-env-vars "GCP_PROJECT=${PROJECT_ID},STREAMLIT_SERVER_CORS_ALLOW_ALL=false,STREAMLIT_SERVER_ENABLE_CORS=true,MODEL_PROFILE=qwen2.5:1.5b,OLLAMA_NUM_THREADS=4,OLLAMA_NUM_CTX=2048,OLLAMA_NUM_PREDICT=1024,OLLAMA_NUM_PARALLEL=1"

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --platform managed --region "${REGION}" --format 'value(status.url)')
echo "============================================================"
echo "[SUCCESS] Service live at: ${SERVICE_URL}"
echo "============================================================"
