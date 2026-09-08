# Local Kubernetes (kind) workflow. See deploy/k8s/README.md.
#
# The Render deploy and docker-compose are unaffected by any of this.

KIND_CLUSTER      ?= fantasy
K8S_NS            ?= fantasy-local
K8S_OVERLAY       ?= deploy/k8s/overlays/local
CORE_IMAGE        ?= fantasy-core:local
ANALYSIS_IMAGE    ?= fantasy-analysis:local
# ingress-nginx release used for kind's provider manifest.
INGRESS_NGINX_REF ?= controller-v1.11.3

KUBECTL := kubectl -n $(K8S_NS)

.DEFAULT_GOAL := help

.PHONY: help
help: ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- one-time cluster lifecycle ------------------------------------------------

.PHONY: cluster-up
cluster-up: ## Create the kind cluster and install ingress-nginx (run once)
	kind create cluster --config deploy/k8s/kind-cluster.yaml
	kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/$(INGRESS_NGINX_REF)/deploy/static/provider/kind/deploy.yaml
	kubectl -n ingress-nginx wait --for=condition=ready pod \
		--selector=app.kubernetes.io/component=controller --timeout=180s

.PHONY: cluster-down
cluster-down: ## Delete the whole kind cluster (wipes all data)
	kind delete cluster --name $(KIND_CLUSTER)

# --- build / load / deploy ----------------------------------------------------

.PHONY: k8s-build
k8s-build: ## Build both service images
	docker build -t $(CORE_IMAGE) -f Dockerfile .
	docker build -t $(ANALYSIS_IMAGE) -f analysis_service/Dockerfile .

.PHONY: k8s-load
k8s-load: k8s-build ## Build, then load both images into the kind node
	kind load docker-image --name $(KIND_CLUSTER) $(CORE_IMAGE) $(ANALYSIS_IMAGE)

.PHONY: k8s-up
k8s-up: k8s-load ## Build + load + apply the local overlay + wait for rollout
	@test -f $(K8S_OVERLAY)/secrets.env || { \
		echo "ERROR: $(K8S_OVERLAY)/secrets.env is missing."; \
		echo "  cp $(K8S_OVERLAY)/secrets.example.env $(K8S_OVERLAY)/secrets.env  and fill it in"; \
		exit 1; }
	kubectl apply -k $(K8S_OVERLAY)
	$(KUBECTL) rollout status statefulset/core-db --timeout=300s
	$(KUBECTL) rollout status statefulset/analysis-db --timeout=300s
	$(KUBECTL) rollout restart deployment/core deployment/analysis
	$(KUBECTL) rollout status deployment/analysis --timeout=300s
	$(KUBECTL) rollout status deployment/core --timeout=300s
	@echo
	@echo "  ready -> http://fantasy.localtest.me/"

.PHONY: k8s-down
k8s-down: ## Remove the app (keeps the namespace and the Postgres PVCs / data)
	$(KUBECTL) delete deployment,statefulset,service,ingress,configmap,secret \
		--all --ignore-not-found
	@echo "namespace + PVCs kept. 'make cluster-down' or 'kubectl delete ns $(K8S_NS)' to wipe data."

.PHONY: k8s-restart
k8s-restart: ## Roll both service Deployments (no rebuild)
	$(KUBECTL) rollout restart deployment/core deployment/analysis

# --- observe ----------------------------------------------------------------

.PHONY: k8s-status
k8s-status: ## Show pods / services / ingress / PVCs
	$(KUBECTL) get pods,svc,ingress,pvc

.PHONY: k8s-logs
k8s-logs: ## Tail logs from every pod in the stack
	$(KUBECTL) logs -l app.kubernetes.io/part-of=fantasy-draft-consultant \
		--all-containers --prefix --tail=100 -f
