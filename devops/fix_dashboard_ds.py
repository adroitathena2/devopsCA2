import json

with open("monitoring/grafana-dashboard.json") as f:
    d = json.load(f)

DS = {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}


def fix(o):
    if isinstance(o, dict):
        ds = o.get("datasource")
        if isinstance(ds, dict) and ds.get("type") == "prometheus":
            o["datasource"] = dict(DS)
        for v in o.values():
            fix(v)
    elif isinstance(o, list):
        for v in o:
            fix(v)


fix(d)
d["__inputs"] = [
    {
        "name": "DS_PROMETHEUS",
        "label": "Prometheus",
        "description": "Prometheus scraping clinical-copilot",
        "type": "datasource",
        "pluginId": "prometheus",
        "pluginName": "Prometheus",
    }
]

with open("monitoring/grafana-dashboard.json", "w") as f:
    json.dump(d, f, indent=2)

print("panels:", len(d["panels"]), "| input:", d["__inputs"][0]["name"])
