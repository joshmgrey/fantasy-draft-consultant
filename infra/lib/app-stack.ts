import {
  CfnOutput,
  Duration,
  RemovalPolicy,
  SecretValue,
  Stack,
  StackProps,
} from 'aws-cdk-lib';
import * as acm from 'aws-cdk-lib/aws-certificatemanager';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as rds from 'aws-cdk-lib/aws-rds';
import * as route53 from 'aws-cdk-lib/aws-route53';
import * as route53Targets from 'aws-cdk-lib/aws-route53-targets';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as servicediscovery from 'aws-cdk-lib/aws-servicediscovery';
import { Construct } from 'constructs';
import {
  ANALYSIS_DB_NAME,
  ANALYSIS_DNS_NAME,
  ANALYSIS_GUNICORN,
  ANALYSIS_REPO_NAME,
  CONTAINER_PORT,
  CORE_DB_NAME,
  CORE_GUNICORN,
  CORE_REPO_NAME,
  PLACEHOLDER,
  RAG_INDEXER_REPO_NAME,
  SECRET_NAMES,
} from './config';

export interface AppStackProps extends StackProps {
  /** Image tag to run for both services (CI passes the git SHA). */
  readonly imageTag: string;
  /** Public hostname, e.g. draft.example.com. Optional. */
  readonly domainName?: string;
  /** Route 53 hosted zone for domainName. Needs hostedZoneName too. */
  readonly hostedZoneId?: string;
  readonly hostedZoneName?: string;
  /**
   * Existing ACM certificate (same region as the stack). If omitted but a
   * domain and hosted zone are given, a DNS-validated certificate is created.
   */
  readonly certificateArn?: string;
  /** Image for the one-off Render -> RDS migration task. */
  readonly migrateImage?: string;
}


/** A health check run inside the container. The images have python, not curl. */
const HEALTHZ_CHECK: ecs.HealthCheck = {
  command: [
    'CMD',
    'python',
    '-c',
    `import urllib.request; urllib.request.urlopen('http://localhost:${CONTAINER_PORT}/healthz', timeout=3)`,
  ],
  interval: Duration.seconds(15),
  timeout: Duration.seconds(5),
  retries: 3,
  startPeriod: Duration.seconds(30),
};

/**
 * Shell for `sh -c`: export each env var as a postgresql:// URL to the named
 * database (built from DB_USERNAME / DB_PASSWORD / DB_HOST / DB_PORT), then
 * exec the real command so gunicorn is PID 1 and gets ECS's SIGTERM.
 */
export function withDatabaseUrls(urls: Record<string, string>, command: string): string {
  const exports = Object.entries(urls).map(
    ([envVar, dbName]) =>
      `export ${envVar}="postgresql://$DB_USERNAME:$DB_PASSWORD@$DB_HOST:$DB_PORT/${dbName}?sslmode=require"`,
  );
  return [...exports, `exec ${command}`].join('; ');
}

/**
 * Creates the analysis database on the shared instance (RDS only creates the
 * one named at launch, `core`) and enables pgvector in it. Runs as a
 * short-lived init container before the analysis app starts. Idempotent, and
 * safe when two tasks start together during a deployment.
 */
export const DB_INIT_SCRIPT = `
import os
import psycopg
from psycopg import errors, sql

conn = dict(
    host=os.environ["DB_HOST"],
    port=os.environ["DB_PORT"],
    user=os.environ["DB_USERNAME"],
    password=os.environ["DB_PASSWORD"],
    sslmode="require",
    connect_timeout=10,
    autocommit=True,
)
name = os.environ["TARGET_DB_NAME"]

with psycopg.connect(dbname="postgres", **conn) as c:
    exists = c.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
    if not exists:
        try:
            c.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
            print(f"created database {name}")
        except errors.DuplicateDatabase:
            pass

with psycopg.connect(dbname=name, **conn) as c:
    c.execute("CREATE EXTENSION IF NOT EXISTS vector")

print(f"database {name} ready (pgvector enabled)")
`.trim();

/**
 * Copies the Render databases into RDS. Run once by hand with
 * `aws ecs run-task` during cutover (see infra/README.md). Dumps to the
 * task's ephemeral disk first, then restores, replacing any tables the apps
 * created on boot.
 */
export const MIGRATE_SCRIPT = `
set -eu
restore() {
  src="$1"; db="$2"
  echo "dumping -> $db"
  pg_dump --format=custom --no-owner --no-acl --file="/tmp/$db.dump" "$src"
  echo "restoring -> $db"
  pg_restore --clean --if-exists --no-owner --no-acl --exit-on-error \\
    --dbname="postgresql://$DB_USERNAME:$DB_PASSWORD@$DB_HOST:$DB_PORT/$db?sslmode=require" "/tmp/$db.dump"
}
restore "$RENDER_DATABASE_URL" ${CORE_DB_NAME}
case "$RENDER_ANALYSIS_DATABASE_URL" in
  postgres*) restore "$RENDER_ANALYSIS_DATABASE_URL" ${ANALYSIS_DB_NAME} ;;
  *) echo "no analysis database to migrate" ;;
esac
echo "migration complete"
`.trim();

/** What the RAG index task runs (also the indexer image's CMD). */
export const RAG_INDEX_COMMAND = 'python -m ml.download --rag-only && python -m ml.rag.index';

export class AppStack extends Stack {
  constructor(scope: Construct, id: string, props: AppStackProps) {
    super(scope, id, props);

    if (Boolean(props.hostedZoneId) !== Boolean(props.hostedZoneName)) {
      throw new Error('hostedZoneId and hostedZoneName must be given together');
    }
    if (props.hostedZoneId && !props.domainName) {
      throw new Error('hostedZoneId needs domainName');
    }

    // ---------------------------------------------------------------- network
    //
    // No NAT gateway. The tasks need outbound internet (ECR, Secrets Manager,
    // CloudWatch Logs, the Anthropic API, Stripe), and the usual answer is
    // private subnets + a NAT gateway: ~$33/month per AZ before data charges,
    // more than the rest of this stack's compute. VPC interface endpoints
    // don't remove it either: they cost ~$7/month each per AZ, we'd need five
    // or six, and api.anthropic.com / api.stripe.com still need the internet.
    //
    // So the Fargate tasks run in the public subnets with a public IP, and the
    // security groups do the isolating instead:
    //   - core accepts traffic only from the ALB's security group;
    //   - analysis accepts traffic only from core's security group;
    //   - egress is limited to HTTPS plus the specific internal hops.
    // A public IP with no inbound rule is unreachable from the internet. The
    // database sits in isolated subnets with no route to the internet at all.
    const vpc = new ec2.Vpc(this, 'Vpc', {
      maxAzs: 2,
      natGateways: 0,
      subnetConfiguration: [
        { name: 'public', subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 },
        { name: 'db', subnetType: ec2.SubnetType.PRIVATE_ISOLATED, cidrMask: 24 },
      ],
    });
    const publicSubnets: ec2.SubnetSelection = { subnetType: ec2.SubnetType.PUBLIC };

    const sg = (logicalId: string, description: string) =>
      new ec2.SecurityGroup(this, logicalId, { vpc, description, allowAllOutbound: false });
    const albSg = sg('AlbSg', 'ALB: HTTP/HTTPS from the internet');
    const coreSg = sg('CoreSg', 'Core app tasks: only the ALB may connect');
    const analysisSg = sg('AnalysisSg', 'Analysis tasks: only core may connect');
    const dbSg = sg('DbSg', 'Postgres: only the app and migration tasks may connect');
    const migrateSg = sg('MigrateSg', 'One-off Render to RDS migration task');

    albSg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(80), 'HTTP (redirects to HTTPS when a cert is set)');
    albSg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(443), 'HTTPS');

    const app = ec2.Port.tcp(CONTAINER_PORT);
    const pg = ec2.Port.tcp(5432);
    const https = ec2.Port.tcp(443);

    albSg.addEgressRule(coreSg, app, 'ALB to core');
    coreSg.addIngressRule(albSg, app, 'ALB to core');

    coreSg.addEgressRule(analysisSg, app, 'core to analysis (Service Connect)');
    analysisSg.addIngressRule(coreSg, app, 'core to analysis (Service Connect)');

    for (const [client, name] of [
      [coreSg, 'core'],
      [analysisSg, 'analysis'],
      [migrateSg, 'migrate'],
    ] as const) {
      client.addEgressRule(dbSg, pg, `${name} to postgres`);
      dbSg.addIngressRule(client, pg, `${name} to postgres`);
      // ECR, Secrets Manager, CloudWatch Logs, SSM (ECS Exec), Anthropic, Stripe.
      client.addEgressRule(ec2.Peer.anyIpv4(), https, 'HTTPS to AWS APIs and third parties');
    }
    // The migration task reads from Render's Postgres over the internet.
    migrateSg.addEgressRule(ec2.Peer.anyIpv4(), pg, 'Render Postgres (external URL)');

    // --------------------------------------------------------------- database
    //
    // One db.t4g.micro instance holding two databases (core, analysis) rather
    // than two instances. The services keep separate databases, so no tables
    // are shared and either can be moved to its own instance later with a
    // pg_dump. But at this traffic a second instance would only double the
    // cost (~$15/month each with storage) and the maintenance windows.
    // The master password is spliced into a postgresql:// URL by a shell
    // one-liner, so keep it alphanumeric: nothing to URL-encode, nothing for
    // the shell to interpret. Built explicitly because
    // Credentials.fromGeneratedSecret only takes excludeCharacters: listing
    // every punctuation character there leaves punctuation an *allowed* type
    // with no characters left, and Secrets Manager (which requires one of each
    // allowed type by default) can't generate the password. excludePunctuation
    // removes the type itself.
    const dbMasterSecret = new secretsmanager.Secret(this, 'DatabaseMasterSecret', {
      description: 'RDS master credentials (username + generated alphanumeric password)',
      generateSecretString: {
        secretStringTemplate: JSON.stringify({ username: 'fdcadmin' }),
        generateStringKey: 'password',
        passwordLength: 30,
        excludePunctuation: true,
      },
    });
    const db = new rds.DatabaseInstance(this, 'Database', {
      // Same major version as the Render database being migrated (18), so
      // pg_dump/pg_restore run 18 -> 18. Minor versions follow
      // autoMinorVersionUpgrade; a later major upgrade needs
      // allowMajorVersionUpgrade, which CloudFormation requires to change it.
      engine: rds.DatabaseInstanceEngine.postgres({
        version: rds.PostgresEngineVersion.of('18.6', '18'),
      }),
      allowMajorVersionUpgrade: true,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.T4G, ec2.InstanceSize.MICRO),
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
      securityGroups: [dbSg],
      credentials: rds.Credentials.fromSecret(dbMasterSecret),
      databaseName: CORE_DB_NAME,
      allocatedStorage: 20,
      maxAllocatedStorage: 100,
      storageType: rds.StorageType.GP3,
      storageEncrypted: true,
      multiAz: false,
      publiclyAccessible: false,
      backupRetention: Duration.days(7),
      deletionProtection: true,
      removalPolicy: RemovalPolicy.SNAPSHOT,
      autoMinorVersionUpgrade: true,
      cloudwatchLogsExports: ['postgresql'],
      cloudwatchLogsRetention: logs.RetentionDays.ONE_MONTH,
    });
    const dbSecret = db.secret!;

    // ---------------------------------------------------------------- secrets
    //
    // Injected as ECS secrets: the task pulls them at start, and they never
    // appear in the task definition or the CloudFormation template.
    // REPLACE_ME placeholders must be overwritten once (infra/README.md).
    // CloudFormation leaves a secret's value alone on later deploys unless
    // this code changes, so a value set with put-secret-value sticks.
    const placeholderSecret = (logicalId: string, secretName: string, description: string, value: SecretValue) =>
      new secretsmanager.Secret(this, logicalId, {
        secretName,
        description,
        secretStringValue: value,
        removalPolicy: RemovalPolicy.RETAIN,
      });
    const generatedSecret = (logicalId: string, secretName: string, description: string) =>
      new secretsmanager.Secret(this, logicalId, {
        secretName,
        description,
        generateSecretString: { passwordLength: 64, excludePunctuation: true },
        removalPolicy: RemovalPolicy.RETAIN,
      });

    const anthropicApiKey = placeholderSecret(
      'AnthropicApiKey',
      SECRET_NAMES.anthropicApiKey,
      'ANTHROPIC_API_KEY for the analysis service',
      SecretValue.unsafePlainText(PLACEHOLDER),
    );
    const flaskSecretKey = generatedSecret('FlaskSecretKey', SECRET_NAMES.secretKey, 'Flask SECRET_KEY for the core app');
    const analysisTokenSecret = generatedSecret(
      'AnalysisTokenSecret',
      SECRET_NAMES.analysisTokenSecret,
      'ANALYSIS_TOKEN_SECRET shared by core and analysis (HMAC actor tokens)',
    );
    const stripe = placeholderSecret(
      'Stripe',
      SECRET_NAMES.stripe,
      'Stripe keys for the core app (JSON)',
      SecretValue.unsafePlainText(
        JSON.stringify({
          STRIPE_SECRET_KEY: PLACEHOLDER,
          STRIPE_PRICE_ID: PLACEHOLDER,
          STRIPE_WEBHOOK_SECRET: PLACEHOLDER,
        }),
      ),
    );
    const renderMigration = placeholderSecret(
      'RenderMigration',
      SECRET_NAMES.renderMigration,
      'Render external database URLs, read only by the one-off migration task',
      SecretValue.unsafePlainText(
        JSON.stringify({ RENDER_DATABASE_URL: PLACEHOLDER, RENDER_ANALYSIS_DATABASE_URL: PLACEHOLDER }),
      ),
    );

    const fromSecret = ecs.Secret.fromSecretsManager;
    const dbEnv = {
      DB_HOST: db.dbInstanceEndpointAddress,
      DB_PORT: db.dbInstanceEndpointPort,
    };
    const dbSecrets = {
      DB_USERNAME: fromSecret(dbSecret, 'username'),
      DB_PASSWORD: fromSecret(dbSecret, 'password'),
    };

    // ------------------------------------------------------------ ECS cluster
    const cluster = new ecs.Cluster(this, 'Cluster', {
      vpc,
      // Service Connect needs a Cloud Map namespace. An HTTP (API-only)
      // namespace is enough and, unlike a DNS one, costs nothing.
      defaultCloudMapNamespace: {
        name: 'fdc.internal',
        type: servicediscovery.NamespaceType.HTTP,
        useForServiceConnect: true,
      },
      containerInsightsV2: ecs.ContainerInsights.DISABLED,
    });

    const logGroup = new logs.LogGroup(this, 'Logs', {
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: RemovalPolicy.DESTROY,
    });
    const logsFor = (streamPrefix: string) => ecs.LogDrivers.awsLogs({ logGroup, streamPrefix });

    const coreRepo = ecr.Repository.fromRepositoryName(this, 'CoreRepo', CORE_REPO_NAME);
    const analysisRepo = ecr.Repository.fromRepositoryName(this, 'AnalysisRepo', ANALYSIS_REPO_NAME);
    const runtimePlatform: ecs.RuntimePlatform = {
      cpuArchitecture: ecs.CpuArchitecture.X86_64,
      operatingSystemFamily: ecs.OperatingSystemFamily.LINUX,
    };

    // The timeouts along the /analyze path, outermost first, each a little
    // longer than the one inside it so the inner one fires first with a
    // useful error: ALB idle 120s > Service Connect 120s >= core's
    // ANALYSIS_READ_TIMEOUT 100s > the analysis service's own Claude calls.
    // gunicorn's worker timeout is 120s in both images.
    const requestTimeout = Duration.seconds(120);

    // --------------------------------------------------------------- analysis
    const analysisTask = new ecs.FargateTaskDefinition(this, 'AnalysisTask', {
      cpu: 512,
      // At least 1 GB: the RAG work bakes a ~130 MB ONNX embedding model into
      // this image and each of the two gunicorn workers loads its own copy.
      // 2 GB leaves headroom for onnxruntime on top of scikit-learn.
      memoryLimitMiB: 2048,
      runtimePlatform,
    });
    const analysisImage = ecs.ContainerImage.fromEcrRepository(analysisRepo, props.imageTag);

    const dbInit = analysisTask.addContainer('db-init', {
      image: analysisImage,
      essential: false,
      entryPoint: ['python', '-c'],
      command: [DB_INIT_SCRIPT],
      environment: { ...dbEnv, TARGET_DB_NAME: ANALYSIS_DB_NAME },
      secrets: dbSecrets,
      logging: logsFor('analysis-db-init'),
    });

    const analysisDbUrl = { ANALYSIS_DATABASE_URL: ANALYSIS_DB_NAME, RAG_DATABASE_URL: ANALYSIS_DB_NAME };
    const analysisContainer = analysisTask.addContainer('analysis', {
      image: analysisImage,
      essential: true,
      entryPoint: ['sh', '-c'],
      // RAG_DATABASE_URL is for the pgvector retrieval work; it shares the
      // analysis database, where db-init has already enabled the extension.
      command: [withDatabaseUrls(analysisDbUrl, ANALYSIS_GUNICORN)],
      environment: { ...dbEnv },
      secrets: {
        ...dbSecrets,
        ANTHROPIC_API_KEY: fromSecret(anthropicApiKey),
        ANALYSIS_TOKEN_SECRET: fromSecret(analysisTokenSecret),
      },
      portMappings: [{ name: 'http', containerPort: CONTAINER_PORT, appProtocol: ecs.AppProtocol.http }],
      healthCheck: HEALTHZ_CHECK,
      logging: logsFor('analysis'),
    });
    analysisContainer.addContainerDependencies({
      container: dbInit,
      condition: ecs.ContainerDependencyCondition.SUCCESS,
    });

    const analysisService = new ecs.FargateService(this, 'AnalysisService', {
      cluster,
      taskDefinition: analysisTask,
      desiredCount: 1,
      minHealthyPercent: 100,
      maxHealthyPercent: 200,
      assignPublicIp: true,
      vpcSubnets: publicSubnets,
      securityGroups: [analysisSg],
      circuitBreaker: { enable: true, rollback: true },
      enableExecuteCommand: true,
      // Private only: no load balancer. Core reaches it as
      // http://analysis:8000 through the Service Connect proxy.
      serviceConnectConfiguration: {
        services: [
          {
            portMappingName: 'http',
            dnsName: ANALYSIS_DNS_NAME,
            port: CONTAINER_PORT,
            // Service Connect's default per-request timeout is 15s; an
            // /analyze call with web search takes far longer.
            perRequestTimeout: requestTimeout,
            idleTimeout: requestTimeout,
          },
        ],
        logDriver: logsFor('analysis-service-connect'),
      },
    });

    // ------------------------------------------------------------------- core
    const coreTask = new ecs.FargateTaskDefinition(this, 'CoreTask', {
      cpu: 256,
      memoryLimitMiB: 512,
      runtimePlatform,
    });
    coreTask.addContainer('core', {
      image: ecs.ContainerImage.fromEcrRepository(coreRepo, props.imageTag),
      essential: true,
      entryPoint: ['sh', '-c'],
      command: [withDatabaseUrls({ DATABASE_URL: CORE_DB_NAME }, CORE_GUNICORN)],
      environment: {
        ...dbEnv,
        ANALYSIS_MODE: 'http',
        ANALYSIS_SERVICE_URL: `http://${ANALYSIS_DNS_NAME}:${CONTAINER_PORT}`,
        ANALYSIS_READ_TIMEOUT: '100',
        // TLS ends at the ALB. gunicorn only trusts X-Forwarded-Proto from
        // these addresses, and the Stripe checkout return URLs are built from
        // request.host_url, so trust the ALB. '*' is safe because coreSg
        // accepts connections from the ALB and nothing else.
        FORWARDED_ALLOW_IPS: '*',
      },
      secrets: {
        ...dbSecrets,
        SECRET_KEY: fromSecret(flaskSecretKey),
        ANALYSIS_TOKEN_SECRET: fromSecret(analysisTokenSecret),
        STRIPE_SECRET_KEY: fromSecret(stripe, 'STRIPE_SECRET_KEY'),
        STRIPE_PRICE_ID: fromSecret(stripe, 'STRIPE_PRICE_ID'),
        STRIPE_WEBHOOK_SECRET: fromSecret(stripe, 'STRIPE_WEBHOOK_SECRET'),
      },
      portMappings: [{ name: 'http', containerPort: CONTAINER_PORT, appProtocol: ecs.AppProtocol.http }],
      healthCheck: HEALTHZ_CHECK,
      logging: logsFor('core'),
    });

    const coreService = new ecs.FargateService(this, 'CoreService', {
      cluster,
      taskDefinition: coreTask,
      desiredCount: 1,
      minHealthyPercent: 100,
      maxHealthyPercent: 200,
      assignPublicIp: true,
      vpcSubnets: publicSubnets,
      securityGroups: [coreSg],
      circuitBreaker: { enable: true, rollback: true },
      enableExecuteCommand: true,
      healthCheckGracePeriod: Duration.seconds(60),
      // Client-only Service Connect: gives core the `analysis` name.
      serviceConnectConfiguration: {
        logDriver: logsFor('core-service-connect'),
      },
    });
    // A Service Connect client only learns endpoints that exist when its tasks
    // start, so bring analysis up first.
    coreService.node.addDependency(analysisService);

    // -------------------------------------------------------------------- ALB
    const alb = new elbv2.ApplicationLoadBalancer(this, 'Alb', {
      vpc,
      internetFacing: true,
      vpcSubnets: publicSubnets,
      securityGroup: albSg,
      idleTimeout: requestTimeout,
      dropInvalidHeaderFields: true,
    });

    const zone =
      props.hostedZoneId && props.hostedZoneName
        ? route53.HostedZone.fromHostedZoneAttributes(this, 'Zone', {
            hostedZoneId: props.hostedZoneId,
            zoneName: props.hostedZoneName,
          })
        : undefined;

    let certificate: acm.ICertificate | undefined;
    if (props.certificateArn) {
      certificate = acm.Certificate.fromCertificateArn(this, 'Certificate', props.certificateArn);
    } else if (props.domainName && zone) {
      certificate = new acm.Certificate(this, 'Certificate', {
        domainName: props.domainName,
        validation: acm.CertificateValidation.fromDns(zone),
      });
    }

    // Defined on its own (not via listener.addTargets) so its logical ID is
    // the same with and without a certificate: adding HTTPS later must not
    // replace the target group the core service is registered with.
    const coreTargetGroup = new elbv2.ApplicationTargetGroup(this, 'CoreTargetGroup', {
      vpc,
      port: CONTAINER_PORT,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targetType: elbv2.TargetType.IP,
      targets: [coreService.loadBalancerTarget({ containerName: 'core', containerPort: CONTAINER_PORT })],
      // Let in-flight /analyze requests (up to ANALYSIS_READ_TIMEOUT=100s)
      // finish on the old task during a rolling deploy instead of being cut
      // off; matches the ALB idle timeout.
      deregistrationDelay: requestTimeout,
      healthCheck: {
        path: '/healthz',
        healthyHttpCodes: '200',
        interval: Duration.seconds(15),
        timeout: Duration.seconds(5),
        healthyThresholdCount: 2,
        unhealthyThresholdCount: 3,
      },
    });

    // One port-80 listener with a fixed logical ID in both modes. Without a
    // certificate it serves the app (first run, on the ALB's DNS name); with
    // one it redirects to HTTPS. Switching modes is then an in-place update
    // of this listener's default action. Two differently-named listeners
    // would make CloudFormation create the new port-80 listener before
    // deleting the old one, which the ALB rejects.
    const httpListener = alb.addListener('Http', {
      port: 80,
      open: false,
      defaultAction: certificate
        ? elbv2.ListenerAction.redirect({ protocol: 'HTTPS', port: '443', permanent: true })
        : elbv2.ListenerAction.forward([coreTargetGroup]),
    });

    const httpsListener = certificate
      ? alb.addListener('Https', {
          port: 443,
          protocol: elbv2.ApplicationProtocol.HTTPS,
          certificates: [certificate],
          sslPolicy: elbv2.SslPolicy.RECOMMENDED_TLS,
          open: false,
          defaultAction: elbv2.ListenerAction.forward([coreTargetGroup]),
        })
      : undefined;

    // The HTTPS listener references the certificate, so it waits for DNS
    // validation. Make the HTTP->HTTPS redirect and the DNS record wait for
    // the HTTPS listener in turn: otherwise either can go live while 443 has
    // nothing behind it.
    if (httpsListener) httpListener.node.addDependency(httpsListener);

    if (zone && props.domainName) {
      const alias = new route53.ARecord(this, 'AliasRecord', {
        zone,
        recordName: props.domainName,
        target: route53.RecordTarget.fromAlias(new route53Targets.LoadBalancerTarget(alb)),
      });
      if (httpsListener) alias.node.addDependency(httpsListener);
    }

    // ------------------------------------------------------ migration (manual)
    // Not a service: a task definition run once with `aws ecs run-task`.
    const migrateTask = new ecs.FargateTaskDefinition(this, 'MigrateTask', {
      cpu: 512,
      memoryLimitMiB: 1024,
      runtimePlatform,
    });
    migrateTask.addContainer('migrate', {
      image: ecs.ContainerImage.fromRegistry(
        props.migrateImage ?? 'public.ecr.aws/docker/library/postgres:18-alpine',
      ),
      entryPoint: ['sh', '-c'],
      command: [MIGRATE_SCRIPT],
      environment: { ...dbEnv },
      secrets: {
        ...dbSecrets,
        RENDER_DATABASE_URL: fromSecret(renderMigration, 'RENDER_DATABASE_URL'),
        RENDER_ANALYSIS_DATABASE_URL: fromSecret(renderMigration, 'RENDER_ANALYSIS_DATABASE_URL'),
      },
      logging: logsFor('migrate'),
    });

    // ------------------------------------------------------- RAG index (manual)
    // Builds the player_seasons pgvector collection in the analysis database
    // that analysis reads through RAG_DATABASE_URL. Run once after the first
    // deploy and again each May after the season rolls over; the rebuild is
    // blue/green, so the running service keeps answering throughout. It runs
    // with the analysis security group: same database access, HTTPS out (for
    // the nflverse downloads from GitHub), no inbound from the internet.
    const ragIndexTask = new ecs.FargateTaskDefinition(this, 'RagIndexTask', {
      // Embedding several thousand documents on CPU: more cores, a faster run.
      // It's billed only for the minutes it runs.
      cpu: 2048,
      memoryLimitMiB: 4096,
      runtimePlatform,
    });
    ragIndexTask.addContainer('rag-index', {
      image: ecs.ContainerImage.fromEcrRepository(
        ecr.Repository.fromRepositoryName(this, 'RagIndexerRepo', RAG_INDEXER_REPO_NAME),
        props.imageTag,
      ),
      entryPoint: ['sh', '-c'],
      command: [
        withDatabaseUrls(
          { RAG_DATABASE_URL: ANALYSIS_DB_NAME },
          `sh -c '${RAG_INDEX_COMMAND}'`,
        ),
      ],
      environment: { ...dbEnv },
      secrets: dbSecrets,
      logging: logsFor('rag-index'),
    });

    // ---------------------------------------------------------------- outputs
    const scheme = certificate ? 'https' : 'http';
    new CfnOutput(this, 'AppUrl', {
      value: `${scheme}://${props.domainName && certificate ? props.domainName : alb.loadBalancerDnsName}`,
    });
    new CfnOutput(this, 'AlbDnsName', { value: alb.loadBalancerDnsName });
    new CfnOutput(this, 'ClusterName', { value: cluster.clusterName });
    new CfnOutput(this, 'CoreServiceName', { value: coreService.serviceName });
    new CfnOutput(this, 'AnalysisServiceName', { value: analysisService.serviceName });
    new CfnOutput(this, 'MigrateTaskDefinitionArn', { value: migrateTask.taskDefinitionArn });
    new CfnOutput(this, 'MigrateSecurityGroupId', { value: migrateSg.securityGroupId });
    new CfnOutput(this, 'RagIndexTaskDefinitionArn', { value: ragIndexTask.taskDefinitionArn });
    new CfnOutput(this, 'AnalysisSecurityGroupId', { value: analysisSg.securityGroupId });
    new CfnOutput(this, 'PublicSubnetIds', {
      value: vpc.publicSubnets.map((s) => s.subnetId).join(','),
    });
    new CfnOutput(this, 'DatabaseEndpoint', { value: db.dbInstanceEndpointAddress });
    new CfnOutput(this, 'DatabaseSecretArn', { value: dbSecret.secretArn });
  }
}
