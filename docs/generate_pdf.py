"""Generate DevOps report PDF with screenshots."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                PageBreak, Table, TableStyle, HRFlowable)
from reportlab.lib import colors

ROOT = os.path.dirname(os.path.abspath(__file__))  # docs/
REPO = os.path.dirname(ROOT)
OUT = os.path.join(ROOT, "Clinical-Copilot-DevOps-Report.pdf")
PAGE_W, PAGE_H = A4
MAX_W = 7.2 * inch

styles = getSampleStyleSheet()
title_s = ParagraphStyle("Title2", parent=styles["Title"], fontSize=22, spaceAfter=6)
h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceBefore=12, spaceAfter=6)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4)
body = ParagraphStyle("Body2", parent=styles["BodyText"], fontSize=10, leading=14)
cap = ParagraphStyle("Cap", parent=styles["BodyText"], fontSize=8.5, leading=11,
                     textColor=colors.HexColor("#444444"), alignment=TA_CENTER)
cell = ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=9, leading=12)
center = ParagraphStyle("Center", parent=styles["BodyText"], fontSize=10, leading=14, alignment=TA_CENTER)

def img(path, caption=None, max_w=MAX_W, max_h=4.5*inch):
    from PIL import Image as PILImage
    if not os.path.exists(path):
        return [Paragraph(f"<i>[missing: {os.path.basename(path)}]</i>", cap)]
    iw, ih = PILImage.open(path).size
    w = min(max_w, iw * 0.6)
    h = w * ih / iw
    if h > max_h:
        h = max_h
        w = h * iw / ih
    items = [Image(path, width=w, height=h)]
    if caption:
        items.append(Paragraph(caption, cap))
        items.append(Spacer(1, 6))
    return items

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=0.55*inch, rightMargin=0.55*inch,
                        topMargin=0.6*inch, bottomMargin=0.6*inch,
                        title="Clinical Copilot - DevOps Report", author="SIT Pune")
S = []
S += [Paragraph("AI Clinical Copilot — DevOps Report", title_s),
      Paragraph("B.Tech Project · Symbiosis Institute of Technology, Pune · Guide: Dr. Ranjeet Bidwe", center),
      Paragraph("Team: Aayush Joshi (23070122008) · Ankush Dutta (23070122032) · Archisha Yadav (23070122041) · Aryan Srivastava (23070122055) &nbsp;|&nbsp; Sep 2026", center),
      Paragraph("Repo: https://github.com/adroitathena2/devopsCA2 · Full ML details in README.md; this report covers DevOps Tasks 1–5 only.", center),
      Spacer(1, 6), HRFlowable(width="100%", thickness=1, color=colors.grey)]

S += [Paragraph("0. What was deployed", h1),
      Paragraph("The production ML pipeline (torch, GLiNER, Granite 278M, 2.4 GB DrugBank, PyQt desktop) is unsuitable for CI/Docker/K8s demos. "
                "A thin contract-compatible FastAPI wrapper was added so the pipeline behaves like the real system without heavy deps.", body),
      Spacer(1, 4),
      Table([[Paragraph("<b>File</b>", cell), Paragraph("<b>Purpose</b>", cell)],
             [Paragraph("<font face='Courier' size=8>service/app.py</font>", cell),
              Paragraph("FastAPI: GET /, /health, /ready, POST /analyze, GET /metrics (Prometheus)", cell)],
             [Paragraph("<font face='Courier' size=8>service/requirements-service.txt</font>", cell),
              Paragraph("Light deps only: fastapi, uvicorn, prometheus_client, pydantic, pytest, httpx", cell)],
             [Paragraph("<font face='Courier' size=8>tests/test_service.py</font>", cell),
              Paragraph("6 pytest cases (health, clean Rx, severe/moderate DDI, metrics)", cell)]],
            colWidths=[2.2*inch, 5.0*inch],
            style=TableStyle([("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                              ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eeeeee"))])),
      Paragraph("POST /analyze uses README examples (Aspirin+Clopidogrel, Metformin+Glimepiride, Dolo 650) with a small pairwise KB. Verified: <b>pytest → 6 passed</b> (docs/pytest-log.txt).", body)]

S += [Paragraph("Task 1 — Deployment Strategy (GitHub Actions)", h1),
      Paragraph("<b>Choice:</b> GitHub Actions over Codefresh/Harness/Spinnaker — native to the repo, free for students, Docker + kubectl ready. "
                "<b>Workflow</b> .github/workflows/ci-cd.yml has 3 jobs: (1) test — setup-python 3.11, pip install, pytest, validate K8s manifests; "
                "(2) build-and-push on main — buildx, Docker Hub login, sha+latest tags, APP_VERSION=$GITHUB_SHA; "
                "(3) deploy-staging — decode KUBECONFIG_STAGING, kubectl set image, rollout status, smoke test /health + POST /analyze.", body)]
S += img(os.path.join(ROOT, "pipeline.png"), "Pipeline diagram (devops/pipeline-diagram.mmd rendered).")
S += img(os.path.join(ROOT, "screenshots", "github-actions-success.png"), "GitHub Actions run #14 — Lint+Unit tests, Build & push, Rolling update all green.")
S += img(os.path.join(ROOT, "screenshots", "pytest-6-passed.png"), "pytest tests/test_service.py — 6 passed (1.20s).", max_h=2.2*inch)

S += [Paragraph("Task 2 — Configuration Management & IaC (Ansible)", h1),
      Paragraph("<b>Files:</b> ansible/inventory.ini, ansible.cfg, playbook.yml (11 tasks + 1 handler, hosts: clinical, become: true). "
                "Installs python3/pip/venv/curl, creates user <font face='Courier'>clinical</font> + /opt/clinical-copilot, copies app.py + requirements, venv pip install, "
                "systemd unit (uvicorn :8000, Restart=always), systemd enable/start when ansible_service_mgr==systemd else nohup fallback, smoke test GET /health (12×5s).<br/>"
                "<b>Real run (WSL, 2026-09-30):</b> ok=12 changed=9 failed=0 skipped=1; curl 127.0.0.1:8000/health → {status:up}; systemctl is-active → active.<br/>"
                "<font face='Courier' size=8>cd ansible &amp;&amp; ansible-playbook -i inventory.ini playbook.yml</font>", body)]
S += img(os.path.join(ROOT, "screenshots", "ansible-run-top.png"), "Ansible run — Gathering Facts through early tasks.")
S += img(os.path.join(ROOT, "screenshots", "ansible-recap.png"), "Ansible recap — ok=11 changed=0 failed=0 skipped=1.", max_h=2.5*inch)

S += [Paragraph("Task 3 — Containerization & Orchestration (Docker + Kubernetes)", h1),
      Paragraph("<b>Files:</b> Dockerfile (python:3.11-slim, HEALTHCHECK /health, ARG/ENV APP_VERSION, copies only service files), .dockerignore, "
                "k8s/deployment.yaml, k8s/service.yaml, k8s/rolling-update-demo.sh. ServiceMonitor lives in monitoring/ (no prometheus-operator CRDs on cluster).<br/>"
                "<b>Runs:</b> docker build -t clinical-copilot:1.0.0 → success; docker run -p 8000:8000 → /health {status:up,version:1.0.0}, /analyze Aspirin+Clopidogrel → 1 Severe, /metrics counters OK; "
                "kubectl apply -f k8s/ → 3/3 pods Running; rolling update to :2.0.0 (set image + APP_VERSION env) with maxSurge=1,maxUnavailable=0; rollout undo → :1.0.0.", body)]
S += img(os.path.join(ROOT, "screenshots", "curl-health-up.png"), "Docker health — curl localhost:8000/health → status up, version 1.0.0.", max_h=2.0*inch)
S += img(os.path.join(ROOT, "screenshots", "k8s-pods-rollout-history.png"), "K8s — kubectl get pods (3/3 Running) + rollout history revisions 2,3.")
S += img(os.path.join(ROOT, "screenshots", "swagger-analyze-request.png"), "Swagger — POST /analyze request body.", max_h=3.8*inch)
S += img(os.path.join(ROOT, "screenshots", "swagger-analyze-empty-response.png"), "Swagger — /analyze 200 response + schema.", max_h=3.8*inch)
S += img(os.path.join(ROOT, "screenshots", "swagger-schema.png"), "Swagger — validation + GET /metrics.", max_h=3.5*inch)

S += [Paragraph("Task 4 — Monitoring & Logging (Prometheus + Grafana)", h1),
      Paragraph("<b>Instrumentation (service/app.py):</b> middleware + /metrics — http_requests_total{method,endpoint,status}, http_request_latency_seconds{endpoint}, "
                "interactions_detected_total, app_errors_total, app_uptime_seconds, app_info{version,service}.<br/>"
                "<b>Files:</b> monitoring/prometheus.yml (scrape app:8000/metrics, 15s), docker-compose.monitoring.yml (app + prom/prometheus:v2.53.0 + grafana/grafana:11.1.0), "
                "grafana-dashboard.json (5 panels: Uptime, req/s, p95, 5xx rate, Interactions/s), servicemonitor.yaml.<br/>"
                "<b>Runs:</b> 3 containers Up (app healthy); Prometheus /-/healthy → Server is Healthy; up{job=clinical-copilot}=1; after 5× POST /analyze, http_requests_total{endpoint=/analyze,status=200}=5; "
                "Grafana /api/health → {database:ok,version:11.1.0}. Live: Prometheus :9090, Grafana :3000 (admin/admin).", body)]
S += img(os.path.join(ROOT, "screenshots", "prometheus-targets-up.png"), "Prometheus Targets — clinical-copilot (1/1 up) + prometheus (1/1 up).")
S += img(os.path.join(ROOT, "screenshots", "grafana-dashboard-live.png"), "Grafana live — Uptime 1031s, Request rate, p95 latency, Interactions/s.")

S += [Paragraph("Task 5 — Reflection: challenges & lessons", h1),
      Paragraph("1) Heavy ML not container/CI-friendly → thin FastAPI wrapper, same /analyze contract. "
                "2) WSL sudo/become friction → NOPASSWD via wsl -u root, green ok=12 failed=0. "
                "3) ServiceMonitor CRD absent → moved to monitoring/, workflow validates deployment+service only. "
                "4) Docker Hub transient DNS → retry + ARG redeclare after FROM. "
                "5) /health version stuck after image update → set both image and env in demo script.<br/>"
                "<b>Lessons:</b> decouple demo service from research pipeline; probes + rollout status + smoke tests before traffic, rollout undo is cheapest incident response; "
                "metrics-first (/metrics day one) makes promotion gates trivial. Next: Loki + Alertmanager SLOs, cosign signing, k6 load gate, HPA on p95, staging/prod namespaces.<br/>"
                "<b>Slides:</b> docs/Clinical-Copilot-DevOps-Report.pptx (5 slides).", body)]
S += img(os.path.join(ROOT, "architecture.png"), "Architecture diagram (from slides).")
S += img(os.path.join(ROOT, "grafana-mock.png"), "Dashboard layout reference (same 4 time-series + stats, real PromQL titles).")

S += [Paragraph("File index (DevOps deliverables)", h1),
      Paragraph("<font face='Courier' size=7.5>.github/workflows/ci-cd.yml<br/>Dockerfile, .dockerignore<br/>"
                "service/{app.py,requirements-service.txt}, tests/test_service.py<br/>"
                "ansible/{inventory.ini,ansible.cfg,playbook.yml}<br/>k8s/{deployment.yaml,service.yaml,rolling-update-demo.sh}<br/>"
                "monitoring/{prometheus.yml,docker-compose.monitoring.yml,grafana-dashboard.json,servicemonitor.yaml}<br/>"
                "devops/{pipeline-diagram.mmd,make_evidence.py,validate_manifests.py}<br/>"
                "docs/{REPORT.md, Clinical-Copilot-DevOps-Report.pdf/.pptx, architecture.png, pipeline.png, grafana-mock.png,<br/>"
                "pytest-log.txt, metrics-sample.txt, docker-health.json, docker-analyze.json, docker-logs.txt,<br/>"
                "k8s-pods.txt, k8s-rollout-history.txt, k8s-image-after-update.txt, k8s-portforward.log, k8s-health-v2.json,<br/>"
                "prometheus-query.json, prometheus-target.json, grafana-health.json,<br/>"
                "screenshots/*.png (11 files)}</font>", body)]

doc.build(S)
print("wrote", OUT, os.path.getsize(OUT))
