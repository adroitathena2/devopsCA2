"""Generate DevOps evidence diagrams + PPTX report (DevOps part only)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs("docs", exist_ok=True)

# ---------- 1. Architecture diagram ----------
fig, ax = plt.subplots(figsize=(12, 6.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_title("Clinical Copilot — DevOps Architecture", fontsize=15, fontweight="bold", pad=12)

def box(ax, x, y, w, h, text, color="#dbeafe", edge="#2563eb"):
    r = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                facecolor=color, edgecolor=edge, linewidth=1.6)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9,
            fontweight="bold", wrap=True)

def arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#111827", lw=1.6))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.15, label, ha="center", fontsize=7.5, color="#374151")

# top row: dev -> CI -> registry
box(ax, 0.3, 4.6, 2.0, 1.2, "Dev push\nGitHub main")
box(ax, 2.9, 4.6, 2.4, 1.2, "GitHub Actions\npytest + dry-run")
box(ax, 5.9, 4.6, 2.4, 1.2, "Docker Hub\nclinical-copilot:sha", color="#fef3c7", edge="#d97706")
box(ax, 8.9, 4.6, 2.7, 1.2, "Ansible\nruntime config", color="#dcfce7", edge="#16a34a")
# bottom row: k8s + monitoring
box(ax, 0.3, 1.2, 2.6, 1.6, "Kubernetes\nDeploy(3 replicas)\nRollingUpdate\nSvc :80 -> :8000", color="#ede9fe", edge="#7c3aed")
box(ax, 3.5, 1.2, 2.6, 1.6, "FastAPI pods\n/analyze /health\n/metrics", color="#e0f2fe", edge="#0284c7")
box(ax, 6.7, 1.2, 2.3, 1.6, "Prometheus :9090\nscrapes /metrics\n15s", color="#fee2e2", edge="#dc2626")
box(ax, 9.6, 1.2, 2.0, 1.6, "Grafana :3000\nuptime/latency\nerror-rate", color="#ffedd5", edge="#ea580c")

arrow(ax, 2.3, 5.2, 2.9, 5.2)
arrow(ax, 5.3, 5.2, 5.9, 5.2)
arrow(ax, 7.1, 4.6, 7.1, 2.8)
arrow(ax, 4.8, 4.6, 4.8, 2.8)
arrow(ax, 2.9, 2.0, 3.5, 2.0)
arrow(ax, 6.1, 2.0, 6.7, 2.0)
arrow(ax, 9.0, 2.0, 9.6, 2.0)
ax.text(6, 0.4, "Full ML pipeline (GLiNER + Granite + DrugBank) wrapped by lightweight FastAPI for DevOps portability",
        ha="center", fontsize=8.5, style="italic", color="#4b5563")
fig.tight_layout()
fig.savefig("docs/architecture.png", dpi=160)
print("wrote docs/architecture.png")

# ---------- 2. Pipeline flow ----------
fig2, ax2 = plt.subplots(figsize=(12, 3.6))
ax2.set_xlim(0, 12)
ax2.set_ylim(0, 3.4)
ax2.axis("off")
ax2.set_title("CI/CD Pipeline Flow (GitHub Actions -> K8s RollingUpdate)", fontsize=13, fontweight="bold")
stages = ["Push\nmain", "pytest\n6 tests", "docker\nbuild+push", "kubectl\nset image", "rollout\nstatus", "smoke\n/health"]
for i, s in enumerate(stages):
    x = 0.4 + i * 1.9
    c = "#bbf7d0" if i < 2 else ("#bfdbfe" if i < 5 else "#fef08a")
    e = "#15803d" if i < 2 else ("#1d4ed8" if i < 5 else "#a16207")
    box(ax2, x, 1.0, 1.6, 1.2, s, color=c, edge=e)
    if i > 0:
        arrow(ax2, x - 0.3, 1.6, x, 1.6)
ax2.text(6, 0.35, "On smoke failure: kubectl rollout undo (rollback). Strategy: maxSurge=1, maxUnavailable=0",
         ha="center", fontsize=8.5, style="italic", color="#4b5563")
fig2.tight_layout()
fig2.savefig("docs/pipeline.png", dpi=160)
print("wrote docs/pipeline.png")

# ---------- 3. Grafana-style mock dashboard (real metric names) ----------
fig3 = plt.figure(figsize=(12, 7))
fig3.suptitle("Clinical Copilot — Service Health (Grafana layout mock, Prometheus metric names)", fontsize=12, fontweight="bold")
t = np.arange(0, 60)
req_rate = 2 + np.sin(t/8) + np.random.default_rng(0).normal(0, 0.2, 60)
lat_p95 = 0.05 + 0.02*np.abs(np.sin(t/10)) + np.random.default_rng(1).normal(0, 0.005, 60)
err_rate = np.clip(np.random.default_rng(2).normal(0.004, 0.003, 60), 0, None)
inter = np.clip(np.random.default_rng(3).normal(0.15, 0.08, 60), 0, None)
axs = fig3.subplots(2, 2)
axs[0,0].plot(t, req_rate, color="#2563eb"); axs[0,0].set_title("Request rate (req/s)\nsum by (endpoint) (rate(http_requests_total[5m]))", fontsize=9)
axs[0,0].set_ylabel("req/s"); axs[0,0].grid(alpha=0.3)
axs[0,1].plot(t, lat_p95, color="#7c3aed"); axs[0,1].set_title("p95 Latency (s)\nhistogram_quantile(0.95, ...http_request_latency_seconds_bucket)", fontsize=9)
axs[0,1].set_ylabel("s"); axs[0,1].grid(alpha=0.3)
axs[1,0].plot(t, err_rate, color="#dc2626"); axs[1,0].set_title("Error rate (5xx)\nsum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(...)", fontsize=9)
axs[1,0].set_ylabel("fraction"); axs[1,0].grid(alpha=0.3)
axs[1,1].plot(t, inter, color="#16a34a"); axs[1,1].set_title("Interactions flagged / sec\nsum(rate(interactions_detected_total[5m]))", fontsize=9)
axs[1,1].grid(alpha=0.3)
for a in axs.flat:
    a.set_xlabel("minutes")
fig3.tight_layout(rect=[0, 0, 1, 0.93])
fig3.savefig("docs/grafana-mock.png", dpi=160)
print("wrote docs/grafana-mock.png")

# ---------- 4. PPTX (5 slides) ----------
from pptx import Presentation
from pptx.util import Inches, Pt

prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)
def add_slide(title, bullets, img=None):
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    tx = slide.shapes.title
    tx.text = title
    left, top, width, height = Inches(0.5), Inches(1.1), Inches(5.8), Inches(5.9)
    tf = slide.shapes.add_textbox(left, top, width, height).text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = "• " + b
        p.level = 0
        for run in p.runs:
            run.font.size = Pt(15)
    if img and os.path.exists(img):
        slide.shapes.add_picture(img, Inches(6.8), Inches(1.1), width=Inches(6.0), height=Inches(5.9))

add_slide("1 — Architecture (DevOps view)",
          ["Service: FastAPI wrapper (app.py) over Clinical Copilot contract",
           "Deploy: Docker image -> K8s Deployment (3 replicas) + ClusterIP Service",
           "Config: Ansible playbook (user, venv, systemd, smoke test)",
           "Observe: Prometheus scrapes /metrics, Grafana dashboard"],
          "docs/architecture.png")
add_slide("2 — Pipeline flow (GitHub Actions)",
          ["Trigger: push to main / PR / manual",
           "Job test: setup-python 3.11, pytest, kubectl dry-run",
           "Job build-and-push: buildx + Docker Hub tags (sha + latest)",
           "Job deploy-staging: set image -> rollout status -> smoke /health + /analyze; fail -> rollout undo"],
          "docs/pipeline.png")
add_slide("3 — Containerization & orchestration",
          ["Dockerfile: python:3.11-slim, HEALTHCHECK /health, APP_VERSION build-arg",
           "Deployment: RollingUpdate (maxSurge=1, maxUnavailable=0), probes, resources",
           "Rolling update: kubectl set image ... :2.0.0 -> rollout status",
           "Rollback: kubectl rollout undo + rollout history (demo script k8s/rolling-update-demo.sh)"],
          "docs/pipeline.png")
add_slide("4 — Monitoring & logging",
          ["Metrics: http_requests_total, http_request_latency_seconds, interactions_detected_total, app_uptime_seconds",
           "Prometheus: scrape app:8000/metrics every 15s (monitoring/prometheus.yml)",
           "Grafana: 5-panel dashboard (uptime, req/s, p95, 5xx rate, interactions/sec)",
           "Run: docker compose -f monitoring/docker-compose.monitoring.yml up"],
          "docs/grafana-mock.png")
add_slide("5 — Challenges & lessons learned",
          ["Heavy ML (torch/GLiNER/Granite) != container-friendly -> thin FastAPI wrapper for DevOps",
           "Windows dev box: no Ansible binary, no Docker daemon/K8s cluster -> validated YAML + pytest + live /metrics",
           "Probes + zero-downtime strategy matter more than replica count for safety-critical API",
           "Next: sign images, gate promotion on load test, add Loki/Alertmanager + SLO alerts"],
          None)

prs.save("docs/Clinical-Copilot-DevOps-Report.pptx")
print("wrote docs/Clinical-Copilot-DevOps-Report.pptx")
