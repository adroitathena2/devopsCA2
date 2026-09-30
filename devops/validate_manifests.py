import yaml

for f in ["k8s/deployment.yaml", "k8s/service.yaml", "monitoring/servicemonitor.yaml"]:
    d = yaml.safe_load(open(f))
    assert d.get("apiVersion") and d.get("kind") and d["metadata"].get("name"), f
    print(f, "->", d["kind"] + "/" + d["metadata"]["name"], "OK")

dep = yaml.safe_load(open("k8s/deployment.yaml"))
assert dep["spec"]["strategy"]["type"] == "RollingUpdate"
c = dep["spec"]["template"]["spec"]["containers"][0]
assert c["ports"][0]["containerPort"] == 8000
assert c["livenessProbe"]["httpGet"]["path"] == "/health"
assert c["readinessProbe"]["httpGet"]["path"] == "/ready"
print("Deployment strategy + probes OK")
