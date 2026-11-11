# AWS deployment (CDK)

AWS CDK (TypeScript) app that runs the draft consultant on ECS Fargate, as a
replacement for the Render setup in the top-level README. Nothing here has
been deployed yet: the stacks synthesize and are unit-tested (`npm test`,
`npx cdk synth`), but the first real `cdk deploy` will be the first time they
meet AWS.

```
                    Internet
                       │  80 → 443 redirect, 443 (ACM cert)
               ┌───────▼────────┐
               │      ALB       │  public subnets, 2 AZs
               └───────┬────────┘  target group health check: GET /healthz
                       │ :8000 (albSg → coreSg only)
          ┌────────────▼─────────────┐
          │ core (Fargate, 0.25/0.5) │  gunicorn run:app, ANALYSIS_MODE=http
          └────────────┬─────────────┘
                       │ http://analysis:8000  (ECS Service Connect)
          ┌────────────▼─────────────┐
          │ analysis (Fargate, 0.5/2)│  no load balancer, coreSg → analysisSg only
          └────────────┬─────────────┘
                       │ 5432
          ┌────────────▼─────────────┐
          │ RDS Postgres 16 (t4g.micro) │  isolated subnets, no internet route
          │  databases: core, analysis  │  pgvector enabled in analysis
          └─────────────────────────────┘
```

## Stacks

| Stack | What | Deployed by |
|---|---|---|
| `FdcGithubOidc` | GitHub OIDC provider + the role Actions assumes | you, once |
| `FdcEcr` | ECR repos `fdc/core`, `fdc/analysis`, `fdc/rag-indexer` | the workflow (before pushing images) |
| `FdcApp` | VPC, ALB, ECS cluster + 2 services, RDS, Secrets Manager, migration and RAG index tasks | the workflow |

The source is in [`lib/`](lib); names shared with the workflow are in
[`lib/config.ts`](lib/config.ts).

## Design choices

**No NAT gateway.** The tasks need outbound internet (ECR, Secrets Manager,
CloudWatch Logs, the Anthropic API, Stripe). A NAT gateway costs ~$33/month
per AZ before data, and VPC endpoints (~$7/month each per AZ, and we'd need
five or six) still wouldn't reach Anthropic or Stripe. So the Fargate tasks
run in the public subnets with public IPs, and the security groups do the
isolating: core accepts only the ALB, analysis accepts only core, egress is
HTTPS plus the specific internal hops. The database is in isolated subnets
with no route out. If you later want private tasks, add a NAT gateway
(`natGateways: 1`), move the services to `PRIVATE_WITH_EGRESS` and drop
`assignPublicIp`.

**One RDS instance, two databases.** The services keep separate databases
(`core`, `analysis`), so no tables are shared and either can move to its own
instance later with a pg_dump. A second instance would double the database
bill (~$14/month each) for no benefit at this traffic. RDS creates `core` at
launch; a short-lived `db-init` container in the analysis task creates
`analysis` and runs `CREATE EXTENSION IF NOT EXISTS vector` in it before the
analysis app starts. Both services connect as the RDS master user; per-service
Postgres roles are a reasonable later hardening.

**Database URLs from secrets.** ECS can't compose a URL out of secret fields,
so each app container starts under `sh -c`, builds `DATABASE_URL` /
`ANALYSIS_DATABASE_URL` / `RAG_DATABASE_URL` from the injected
`DB_USERNAME`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, then `exec`s the same
gunicorn command as the Dockerfile's `CMD` (kept in `lib/config.ts`; update it
if the Dockerfiles change). The generated password is alphanumeric so it needs
no URL-encoding. Connections use `sslmode=require` (RDS for PG16 forces SSL).

**Timeouts.** ALB idle timeout 120s and a Service Connect per-request timeout
of 120s (its default is 15s, far too short for `/analyze`), above core's
`ANALYSIS_READ_TIMEOUT=100`. `FORWARDED_ALLOW_IPS=*` lets gunicorn trust the
ALB's `X-Forwarded-Proto`, so `request.host_url` (used for the Stripe checkout
return URLs) is `https://`.

**Sizing.** Core 0.25 vCPU / 512 MB. Analysis 0.5 vCPU / 2 GB: the RAG work
bakes a ~130 MB ONNX embedding model into the image and each of the two
gunicorn workers loads it.

## Prerequisites

- Node 22+, Docker, AWS CLI v2, and credentials for the target account
  (`aws sso login` or a profile).
- `cd infra && npm ci`

Local checks (no AWS access needed):

```bash
npm test
```

```bash
npx cdk synth --quiet
```

## One-time bootstrap

1. **Bootstrap CDK** in the account and region (creates the `cdk-*` roles the
   deploy role assumes):

   ```bash
   npx cdk bootstrap aws://ACCOUNT_ID/REGION
   ```

2. **Create the GitHub deploy role.** An account can hold only one GitHub
   OIDC provider. If another project already created it, pass its ARN:

   ```bash
   npx cdk deploy FdcGithubOidc
   ```

   ```bash
   npx cdk deploy FdcGithubOidc -c githubOidcProviderArn=arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com
   ```

   The role trusts only workflow runs on `main` of
   `joshmgrey/fantasy-draft-consultant` (change `githubRepo` in `cdk.json` for
   a fork). Note the `DeployRoleArn` output.

3. **Set the repository variables** (variables, not secrets: none of these is
   sensitive). The deploy workflow is skipped until the first two exist.

   ```bash
   gh variable set AWS_DEPLOY_ROLE_ARN --body "arn:aws:iam::ACCOUNT_ID:role/FdcGithubOidc-DeployRole..."
   ```

   ```bash
   gh variable set AWS_REGION --body "us-east-1"
   ```

4. **Domain (optional; skip for a first run on the ALB's DNS name).**
   - DNS in Route 53: set `DOMAIN_NAME` (e.g. `draft.example.com`),
     `HOSTED_ZONE_ID` and `HOSTED_ZONE_NAME` (e.g. `example.com`). The stack
     creates a DNS-validated ACM certificate and an alias record. Creating the
     record is the DNS cutover, so leave these unset until cutover day if the
     name currently points at Render.
   - DNS elsewhere: request an ACM certificate for the name in the same region
     and validate it, then set `DOMAIN_NAME` and `CERTIFICATE_ARN`. You'll add
     a CNAME to the ALB yourself at cutover.

   Without a certificate the ALB serves plain HTTP on port 80.

5. **First deploy.** Actions → *Deploy to AWS* → *Run workflow* (or push to
   `main`). It tests the stacks, deploys `FdcEcr`, builds and pushes the
   images (core, analysis, rag-indexer) tagged with the commit SHA and `latest`, then deploys `FdcApp`. The
   first `FdcApp` deploy takes ~15–20 minutes, mostly RDS.

   The services come up healthy with placeholder secrets, but `/analyze` and
   billing won't work until the next step.

## Secrets to fill in

All are in Secrets Manager and reach the containers as ECS secrets, never as
plain environment variables or template values.

| Secret | Holds | Initial value | Action |
|---|---|---|---|
| `fdc/anthropic-api-key` | `ANTHROPIC_API_KEY` (analysis only) | `REPLACE_ME` | **set it** |
| `fdc/stripe` | JSON: `STRIPE_SECRET_KEY`, `STRIPE_PRICE_ID`, `STRIPE_WEBHOOK_SECRET` (core only) | `REPLACE_ME`s | **set it** (webhook secret at cutover, below) |
| `fdc/secret-key` | Flask `SECRET_KEY` | random | optional: copy Render's value to keep users logged in across the move |
| `fdc/analysis-token-secret` | `ANALYSIS_TOKEN_SECRET` (both services) | random | leave it |
| `fdc/render-migration` | JSON: Render database URLs, read only by the migration task | `REPLACE_ME`s | at migration time |
| RDS master credentials (`DatabaseSecretArn` output) | `username`, `password`, host, port | random | leave it |

Write values from files so they stay out of shell history:

```bash
aws secretsmanager put-secret-value --secret-id fdc/anthropic-api-key --secret-string file://anthropic.txt
```

```bash
aws secretsmanager put-secret-value --secret-id fdc/stripe --secret-string file://stripe.json
```

with `stripe.json` like `{"STRIPE_SECRET_KEY":"sk_live_...","STRIPE_PRICE_ID":"price_...","STRIPE_WEBHOOK_SECRET":"whsec_..."}`.
Delete the files afterwards. Tasks read secrets when they start, so roll the
services to pick up new values (cluster and service names are stack outputs):

```bash
aws ecs update-service --cluster CLUSTER --service CORE_SERVICE --force-new-deployment
```

```bash
aws ecs update-service --cluster CLUSTER --service ANALYSIS_SERVICE --force-new-deployment
```

CloudFormation only rewrites a secret's value when the code that defines it
changes, so values you set this way survive later deploys.

## Migrating data from Render Postgres

The stack includes a migration task definition (Postgres 16 client image)
that runs `pg_dump --format=custom` against Render and
`pg_restore --clean --if-exists --no-owner --no-acl` into RDS, from inside the
VPC (RDS has no public endpoint). The script is `MIGRATE_SCRIPT` in
[`lib/app-stack.ts`](lib/app-stack.ts).

1. **Check versions.** Render's Postgres version is on the database page. The
   client must be at least that version. For Render on PG17, redeploy with
   `-c migrateImage=public.ecr.aws/docker/library/postgres:17-alpine`
   (pg_restore 17 into a PG16 server works for plain schemas like this one).
2. **Allow access.** In the Render database's *Access Control*, allow
   connections from outside Render (the task's public IP isn't known in
   advance, so `0.0.0.0/0` for the migration window, then remove it).
3. **Freeze writes.** Suspend the Render web service(s) so nothing changes
   after the dump.
4. **Store the source URLs.** Use Render's *External Database URL*s. If Render
   runs the monolith there is no analysis database (its only table, `user`,
   is in the core one); put `none` for the second URL.

   ```bash
   aws secretsmanager put-secret-value --secret-id fdc/render-migration --secret-string file://render.json
   ```

   with `render.json` like `{"RENDER_DATABASE_URL":"postgresql://...render.com/...","RENDER_ANALYSIS_DATABASE_URL":"none"}`.
5. **Run it** (values from the `FdcApp` stack outputs):

   ```bash
   aws ecs run-task --cluster CLUSTER --launch-type FARGATE --task-definition MIGRATE_TASK_DEFINITION_ARN --network-configuration "awsvpcConfiguration={subnets=[SUBNET_1,SUBNET_2],securityGroups=[MIGRATE_SG],assignPublicIp=ENABLED}"
   ```

   Follow it in CloudWatch Logs (log group in the stack's resources, stream
   prefix `migrate`); it ends with `migration complete`, or the task exits
   non-zero on the first error. `--clean` replaces the empty tables the apps
   created on boot, so the run is safe to repeat.
6. **Verify and restart.** Roll both services (commands above) and check that
   an existing user can log in on the ALB URL.
7. **Clean up.** Overwrite `fdc/render-migration` with placeholders and remove
   the Render access-control rule.

## Building the RAG index

The analysis service's retrieval reads the `player_seasons` pgvector
collection in the `analysis` database (`RAG_DATABASE_URL`). Retrieval is
optional, so the app works without the collection, but it adds nothing until
the collection is built. The index isn't migrated from Render: it's rebuilt
from nflverse data.

`FdcApp` has a task definition for this, separate from the services. It runs
the `fdc/rag-indexer` image ([`deploy/rag-indexer/Dockerfile`](../deploy/rag-indexer/Dockerfile):
the analysis dependencies and embedding model, plus pandas and `ml/`). The
task runs `python -m ml.download && python -m ml.rag.index` against the
analysis database. The rebuild is blue/green (staging collection, then one
swap transaction), so the running service keeps answering throughout and
picks up the new collection without a restart.

Run it after the first deploy, and again **each May** after the season rolls
over (values from the `FdcApp` stack outputs):

```bash
aws ecs run-task --cluster CLUSTER --launch-type FARGATE --task-definition RAG_INDEX_TASK_DEFINITION_ARN --network-configuration "awsvpcConfiguration={subnets=[SUBNET_1,SUBNET_2],securityGroups=[ANALYSIS_SG],assignPublicIp=ENABLED}"
```

It runs with the analysis security group (database access plus HTTPS out to
GitHub for the downloads). Follow it in CloudWatch Logs under stream prefix
`rag-index`; it ends with `indexed N player-season documents`. At 2 vCPU /
4 GB it costs a few cents per run. The task definition points at the image
of the last deploy, so the index is always built with the same code and
embedding model as the running analysis service.

## Cutover

1. Bootstrap, deploy, fill secrets, build the RAG index and run a test
   migration ahead of time; check the app on the ALB DNS name (`AlbDnsName` output).
2. Lower the TTL on the app's DNS record (e.g. to 60s) a day before.
3. On the day: suspend Render, run the migration again (fresh data), roll the
   services.
4. Point DNS at AWS: either set `DOMAIN_NAME` / `HOSTED_ZONE_ID` /
   `HOSTED_ZONE_NAME` and re-run the workflow (alias record + cert), or create
   a CNAME to the ALB at your DNS provider (with `CERTIFICATE_ARN` set).
5. **Stripe webhook.** In the Stripe dashboard (Developers → Webhooks), add an
   endpoint `https://NEW_DOMAIN/webhook` for `checkout.session.completed`.
   Each endpoint has its own signing secret: copy the new `whsec_...` into
   `fdc/stripe`'s `STRIPE_WEBHOOK_SECRET` and roll core. Disable the old
   Render endpoint once you see the new one deliver (keep it rather than
   deleting it until you're done with rollback).
6. Run a test checkout end to end; watch the core logs for the webhook.
7. Raise the DNS TTL back.

## Rolling back

- **Bad release on AWS:** a deployment that fails health checks trips the
  ECS circuit breaker and rolls back by itself. To go back to an older image
  deliberately, re-run the workflow on the older commit, or:

  ```bash
  npx cdk deploy FdcApp -c imageTag=OLDER_GIT_SHA
  ```

  ECR keeps the 30 most recent images per repo.
- **Back to Render:** Render keeps working throughout as long as you don't
  delete it. Resume the Render service(s), point DNS back at Render, and
  re-enable the Render Stripe webhook endpoint (and disable the AWS one).
  Data written on AWS after cutover isn't on Render: if it matters, dump it
  back with the same pg_dump/pg_restore pair in the other direction (e.g. from
  an ECS Exec session in a core task, or a copy of the migration task with the
  source and target swapped). Keep Render paid up until you're past the point
  of rolling back.
- **Tearing down AWS:** `npx cdk destroy FdcApp` stops the billing for compute
  and the ALB. The database has deletion protection and is snapshotted on
  delete, and the secrets and ECR repos are retained; remove them by hand if
  you really mean it.

## Rough monthly cost

Approximate us-east-1 on-demand prices, always-on, light traffic, no free
tier. **Check the [AWS Pricing Calculator](https://calculator.aws/) for your
region and current prices before relying on these.**

| Item | Est. $/month |
|---|---|
| Fargate core (0.25 vCPU, 0.5 GB) | ~9 |
| Fargate analysis (0.5 vCPU, 2 GB) | ~21 |
| ALB (hours + light LCU) | ~18 |
| RDS db.t4g.micro single-AZ + 20 GB gp3 | ~14 |
| Public IPv4 addresses (2 ALB + 2 tasks, $0.005/h each) | ~15 |
| Secrets Manager (7 secrets) | ~3 |
| CloudWatch Logs, ECR storage | ~2–5 |
| Route 53 hosted zone (if used) | 0.50 |
| **Total** | **~$80–90** |

Not included: data transfer out, Anthropic API usage, Stripe fees. Levers if
it needs to be cheaper: ARM64/Graviton tasks (~20% off Fargate; build the
images for `linux/arm64`), Fargate Spot for analysis, 1 GB for analysis if the
RAG model isn't in the image, or folding the analysis service back into core
(`ANALYSIS_MODE=inprocess`, one service).
