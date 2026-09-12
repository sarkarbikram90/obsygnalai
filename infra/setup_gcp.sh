#!/usr/bin/env bash
set -euo pipefail

# Ensure gcloud is in PATH if installed in user local appdata
if ! command -v gcloud &>/dev/null; then
  export PATH="${PATH}:/c/Users/bikrams/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin"
fi

echo "============================================================"
echo " Obsygnal AI: GCP Infrastructure & Identity Provisioning"
echo " Target Domain: obsygnal.com"
echo "============================================================"

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
SERVICE_NAME="obsygnal"
CUSTOM_DOMAIN="obsygnal.com"
REGION="${GCP_REGION:-us-central1}"

if [ -z "${PROJECT_ID}" ] || [ "${PROJECT_ID}" = "(unset)" ]; then
  echo "[ERROR] No active GCP project found in gcloud context."
  echo "Please set an active project using: gcloud config set project <PROJECT_ID>"
  exit 1
fi

echo "[INFO] Project ID:     ${PROJECT_ID}"
echo "[INFO] Service Name:   ${SERVICE_NAME}"
echo "[INFO] Custom Domain:  ${CUSTOM_DOMAIN}"
echo "[INFO] Region:         ${REGION}"

echo "[STEP 1/5] Enabling Datastore/Firestore, Cloud Run, and Cloud Build APIs..."
gcloud services enable \
  firestore.googleapis.com \
  datastore.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com

echo "[STEP 2/5] Binding roles/datastore.user to Cloud Run compute service account..."
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "[INFO] Compute Service Account: ${COMPUTE_SA}"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/datastore.user"

echo "[STEP 3/5] Deploying Firestore composite indexes for 'chat_histories'..."
gcloud firestore indexes composite create \
  --collection-group="chat_histories" \
  --field-config field-path="user_id",order=ascending \
  --field-config field-path="updated_at",order=descending || echo "[INFO] Composite index already exists or creation initiated."

echo "[STEP 4/5] Enabling automated Firestore TTL policy on 'expire_at' field..."
gcloud firestore fields ttls update "expire_at" \
  --collection-group="chat_histories" \
  --enable-ttl || echo "[INFO] TTL policy already active or pending."

echo "[STEP 5/5] Mapping custom domain '${CUSTOM_DOMAIN}' to Cloud Run service..."
gcloud beta run domain-mappings create \
  --service="${SERVICE_NAME}" \
  --domain="${CUSTOM_DOMAIN}" \
  --region="${REGION}" || echo "[INFO] Domain mapping command evaluated."

echo "============================================================"
echo "[SUCCESS] Infrastructure setup complete."
echo "Inspect DNS records configured for '${CUSTOM_DOMAIN}':"
echo "  gcloud beta run domain-mappings describe --domain=${CUSTOM_DOMAIN} --region=${REGION}"
echo "============================================================"
