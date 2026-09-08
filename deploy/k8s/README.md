# Local Kubernetes (kind)

Runs the full split stack — core + analysis + a Postgres each — on a local
[kind](https://kind.sigs.k8s.io/) cluster, reachable in the browser through an
ingress-nginx Ingress.

This is a **local development** setup, not production. The Render deploy and
`docker compose up` are unaffected — nothing here changes how those work.

```
deploy/k8s/
├── kind-cluster.yaml          # 1-node cluster: ingress-ready label + host ports 80/443
├── base/                      # environment-agnostic manifests
│   ├── namespace.yaml
│   ├── configmap.yaml         # app-config: ANALYSIS_MODE, service URL, timeouts
│   ├── core-deployment.yaml   core-service.yaml
│   ├── analysis-deployment.yaml   analysis-service.yaml
│   ├── core-db-statefulset.yaml    core-db-service.yaml
│   ├── analysis-db-statefulset.yaml   analysis-db-service.yaml
│   └── ingress.yaml           # nginx, host fantasy.localtest.me -> core-service
└── overlays/local/            # what `make k8s-up` applies
    ├── kustomization.yaml     # base + image :local tags + the Secret
    ├── secrets.example.env    # copy to secrets.env (gitignored) and fill in
    └── secrets.env            # you create this; never committed
```

## Prerequisites

- Docker
- [`kind`](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
- `kubectl`
- `make` (Windows: `choco install make`, or run the target commands by hand)

## One-time: create the cluster

```bash
make cluster-up
```

This creates a single-node kind cluster named `fantasy` (control-plane node
labelled `ingress-ready=true`, host ports 80/443 published) and installs
ingress-nginx using its official kind provider manifest, then waits for the
controller to be ready.

You only do this once. It survives reboots. `make cluster-down` deletes it.

## Bring the stack up

```bash
cd deploy/k8s/overlays/local
cp secrets.example.env secrets.env      # then edit secrets.env
cd -

make k8s-up
```

`secrets.env` needs:

| key | notes |
|---|---|
| `ANALYSIS_TOKEN_SECRET` | shared HMAC secret; both services read this one key, so they match by construction |
| `SECRET_KEY` | Flask session key for core |
| `ANTHROPIC_API_KEY` | analysis service only. Leave **empty** to run without live analysis — the service still starts, `/healthz` reports `anthropic_configured=false`, and `/analyze` returns a clean "not configured" error |
| `CORE_DB_PASSWORD`, `ANALYSIS_DB_PASSWORD` | any non-empty strings |

Generate values: `python -c "import secrets; print(secrets.token_urlsafe(32))"`

`make k8s-up` builds both images, loads them into the kind node
(`kind load docker-image`), applies the `local` overlay, restarts the two
service Deployments so the freshly built images are picked up, and waits for
the rollout.

## Reach it

<http://fantasy.localtest.me/>

`fantasy.localtest.me` resolves to `127.0.0.1` via public DNS, so no hosts-file
edit is needed. Requests hit ingress-nginx on port 80 and are routed to
`core-service`. Sign up for an account and analyze a player.

The analysis service has **no** Ingress — it is only reachable in-cluster, at
`http://analysis-service:8000`, which is exactly how the core app calls it
(`ANALYSIS_MODE=http`).

## Watch it

```bash
make k8s-status                     # pods, services, ingress, PVCs
make k8s-logs                       # tail every pod in the stack

kubectl -n fantasy-local get pods -w
kubectl -n fantasy-local logs -f deployment/core
kubectl -n fantasy-local logs -f deployment/analysis
kubectl -n fantasy-local describe pod <pod>
```

## Roll a deployment

```bash
make k8s-restart                              # both services
kubectl -n fantasy-local rollout restart deployment/analysis
kubectl -n fantasy-local rollout status deployment/analysis
kubectl -n fantasy-local rollout undo deployment/analysis
```

After changing app code, `make k8s-up` again — it rebuilds, reloads, and rolls.

## Try the "analysis service down" path

The core app is designed to degrade gracefully when the analysis service is
unreachable (503 + a "busy, try again" message, and the request is **not**
counted against the user's quota). To see it under k8s:

```bash
kubectl -n fantasy-local scale deployment/analysis --replicas=0
```

`analysis-service` now has zero endpoints. The browser UI still loads, login
still works, and `/analyze` returns the 503 busy message immediately (kube-proxy
rejects the connection — no hang). Bring it back:

```bash
kubectl -n fantasy-local scale deployment/analysis --replicas=1
```

The same thing happens on its own if the analysis pod fails its readiness probe
(`/readyz`, which checks `analysis-db`): it drops out of the Service and the
core app takes the degraded path, without the core pod itself restarting.

## Tear down

```bash
make k8s-down       # removes the app; keeps the namespace and the Postgres PVCs (data)
make cluster-down   # deletes the whole cluster
```

`make k8s-down` then `make k8s-up` brings the stack back with its data intact —
the StatefulSet PVCs (`data-core-db-0`, `data-analysis-db-0`) are retained and
reattached. To wipe data without deleting the cluster:
`kubectl delete ns fantasy-local`.

## Notes / choices

- **Postgres as a StatefulSet** (one replica each) rather than Deployment + PVC:
  the `volumeClaimTemplate` PVC is retained across `make k8s-down`, and a
  rolling update can never briefly double-mount the `ReadWriteOnce` volume.
- **`db.create_all()` at startup** handles schema — there is no migration Job.
  Each service has a `wait-for-db` initContainer so it blocks on its database
  instead of crash-looping.
- **Secret has no name-hash suffix** (`disableNameSuffixHash: true`) so the
  base's `secretKeyRef`s resolve; `make k8s-up` restarts the workloads after
  apply, so a changed secret still takes effect.
- **Stripe is not configured** — billing routes (`/subscribe`, `/webhook`) will
  error, same as `docker compose` without Stripe vars.
