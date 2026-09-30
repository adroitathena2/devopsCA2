#!/bin/bash
# Task 3 — Rolling update & rollback demo (K8s)
set -euo pipefail
NS="${NS:-default}"

echo "==> 1. Apply base manifests (v1.0.0)"
kubectl apply -n "$NS" -f k8s/
kubectl rollout status deployment/clinical-copilot -n "$NS" --timeout=180s

echo "==> 2. Rolling update to v2.0.0 (maxSurge=1, maxUnavailable=0)"
kubectl set image deployment/clinical-copilot \
  clinical-copilot=clinical-copilot:2.0.0 -n "$NS"
kubectl rollout status deployment/clinical-copilot -n "$NS" --timeout=180s

echo "==> 3. Verify new version serving traffic"
kubectl port-forward svc/clinical-copilot 8000:80 -n "$NS" >/tmp/pf.log 2>&1 &
PF_PID=$!
sleep 6
curl -s http://localhost:8000/health
kill $PF_PID || true

echo "==> 4. Rollback to previous revision"
kubectl rollout undo deployment/clinical-copilot -n "$NS"
kubectl rollout status deployment/clinical-copilot -n "$NS" --timeout=180s
kubectl rollout history deployment/clinical-copilot -n "$NS"
echo "Demo complete."
