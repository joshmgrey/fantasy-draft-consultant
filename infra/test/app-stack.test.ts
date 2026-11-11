import { App } from 'aws-cdk-lib';
import { Match, Template } from 'aws-cdk-lib/assertions';
import { AppStack, AppStackProps, RAG_INDEX_COMMAND, withDatabaseUrls } from '../lib/app-stack';

const SENSITIVE_VARS = [
  'ANTHROPIC_API_KEY',
  'SECRET_KEY',
  'ANALYSIS_TOKEN_SECRET',
  'STRIPE_SECRET_KEY',
  'STRIPE_PRICE_ID',
  'STRIPE_WEBHOOK_SECRET',
  'DB_PASSWORD',
  'DATABASE_URL',
  'ANALYSIS_DATABASE_URL',
  'RAG_DATABASE_URL',
];

function synth(overrides: Partial<AppStackProps> = {}): Template {
  const app = new App();
  const stack = new AppStack(app, 'TestApp', { imageTag: 'abc123', ...overrides });
  return Template.fromStack(stack);
}

type Container = {
  Name: string;
  Command?: string[];
  EntryPoint?: string[];
  Environment?: { Name: string; Value: unknown }[];
  Secrets?: { Name: string; ValueFrom: unknown }[];
  HealthCheck?: { Command: string[] };
  DependsOn?: { ContainerName: string; Condition: string }[];
  Essential?: boolean;
  Image: unknown;
};

function containers(template: Template): Record<string, Container> {
  const out: Record<string, Container> = {};
  for (const td of Object.values(template.findResources('AWS::ECS::TaskDefinition'))) {
    for (const c of td.Properties.ContainerDefinitions as Container[]) out[c.Name] = c;
  }
  return out;
}

function taskDefFor(template: Template, containerName: string) {
  return Object.values(template.findResources('AWS::ECS::TaskDefinition')).find((td) =>
    (td.Properties.ContainerDefinitions as Container[]).some((c) => c.Name === containerName),
  )!;
}

const envOf = (c: Container) => Object.fromEntries((c.Environment ?? []).map((e) => [e.Name, e.Value]));
const secretNames = (c: Container) => (c.Secrets ?? []).map((s) => s.Name);

function logicalIdOf(template: Template, type: string, descriptionPrefix: string): string {
  const match = Object.entries(template.findResources(type)).find(([, r]) =>
    String(r.Properties.GroupDescription).startsWith(descriptionPrefix),
  );
  if (!match) throw new Error(`no ${type} described "${descriptionPrefix}..."`);
  return match[0];
}

describe('network', () => {
  const template = synth();

  test('two AZs, public + isolated subnets, no NAT gateway', () => {
    template.resourceCountIs('AWS::EC2::VPC', 1);
    template.resourceCountIs('AWS::EC2::Subnet', 4);
    template.resourceCountIs('AWS::EC2::NatGateway', 0);
    template.resourceCountIs('AWS::EC2::VPCEndpoint', 0);
  });

  test('core only accepts traffic from the ALB, analysis only from core', () => {
    const albSg = logicalIdOf(template, 'AWS::EC2::SecurityGroup', 'ALB');
    const coreSg = logicalIdOf(template, 'AWS::EC2::SecurityGroup', 'Core app');
    const analysisSg = logicalIdOf(template, 'AWS::EC2::SecurityGroup', 'Analysis');

    const ingressTo = (groupId: string) =>
      Object.values(template.findResources('AWS::EC2::SecurityGroupIngress')).filter(
        (r) => r.Properties.GroupId['Fn::GetAtt'][0] === groupId,
      );

    const coreIngress = ingressTo(coreSg);
    expect(coreIngress).toHaveLength(1);
    expect(coreIngress[0].Properties.SourceSecurityGroupId['Fn::GetAtt'][0]).toBe(albSg);
    expect(coreIngress[0].Properties.FromPort).toBe(8000);

    const analysisIngress = ingressTo(analysisSg);
    expect(analysisIngress).toHaveLength(1);
    expect(analysisIngress[0].Properties.SourceSecurityGroupId['Fn::GetAtt'][0]).toBe(coreSg);

    // No inline (CIDR) ingress on either service's group.
    for (const id of [coreSg, analysisSg]) {
      const group = template.toJSON().Resources[id];
      expect(group.Properties.SecurityGroupIngress).toBeUndefined();
    }
  });

  test('app tasks run in public subnets with a public IP (no NAT)', () => {
    template.allResourcesProperties('AWS::ECS::Service', {
      NetworkConfiguration: { AwsvpcConfiguration: { AssignPublicIp: 'ENABLED' } },
    });
  });
});

describe('database', () => {
  const template = synth();

  test('one private, encrypted Postgres 18 instance', () => {
    template.resourceCountIs('AWS::RDS::DBInstance', 1);
    template.hasResource('AWS::RDS::DBInstance', {
      Properties: Match.objectLike({
        Engine: 'postgres',
        EngineVersion: Match.stringLikeRegexp('^18'),
        AllowMajorVersionUpgrade: true,
        DBInstanceClass: 'db.t4g.micro',
        DBName: 'core',
        PubliclyAccessible: false,
        StorageEncrypted: true,
        DeletionProtection: true,
      }),
      DeletionPolicy: 'Snapshot',
    });
  });

  test('analysis database is bootstrapped (with pgvector) before the analysis app starts', () => {
    const c = containers(template);
    expect(c['db-init'].Essential).toBe(false);
    expect(c['db-init'].Command!.join(' ')).toContain('CREATE EXTENSION IF NOT EXISTS vector');
    expect(envOf(c['db-init']).TARGET_DB_NAME).toBe('analysis');
    expect(c.analysis.DependsOn).toEqual([{ ContainerName: 'db-init', Condition: 'SUCCESS' }]);
  });
});

describe('secrets', () => {
  const template = synth();

  test.each([
    'fdc/anthropic-api-key',
    'fdc/secret-key',
    'fdc/analysis-token-secret',
    'fdc/stripe',
    'fdc/render-migration',
  ])('%s exists and is retained on stack deletion', (name) => {
    template.hasResource('AWS::SecretsManager::Secret', {
      Properties: Match.objectLike({ Name: name }),
      DeletionPolicy: 'Retain',
    });
  });

  test('every generated secret can satisfy its own character rules', () => {
    // Secrets Manager requires one character of each type that isn't excluded
    // (RequireEachIncludedType defaults to true). A type left allowed while
    // ExcludeCharacters removes all of its characters makes generation fail
    // at deploy time, so check every generated secret in the template.
    const classes: [string, string][] = [
      ['ExcludeLowercase', 'abcdefghijklmnopqrstuvwxyz'],
      ['ExcludeUppercase', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'],
      ['ExcludeNumbers', '0123456789'],
      ['ExcludePunctuation', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'],
    ];
    const secrets = template.findResources('AWS::SecretsManager::Secret');
    const generated = Object.values(secrets)
      .map((r) => r.Properties?.GenerateSecretString)
      .filter((g) => g !== undefined);
    expect(generated.length).toBeGreaterThanOrEqual(3);
    for (const g of generated) {
      if (g.RequireEachIncludedType === false) continue;
      const excluded = new Set<string>(g.ExcludeCharacters ?? '');
      for (const [flag, chars] of classes) {
        if (g[flag]) continue;
        const left = [...chars].filter((ch) => !excluded.has(ch));
        expect({ flag, left: left.length > 0 }).toEqual({ flag, left: true });
      }
    }
  });

  test('the database master password is alphanumeric', () => {
    template.hasResourceProperties('AWS::SecretsManager::Secret', {
      GenerateSecretString: Match.objectLike({
        GenerateStringKey: 'password',
        SecretStringTemplate: JSON.stringify({ username: 'fdcadmin' }),
        ExcludePunctuation: true,
      }),
    });
  });

  test('sensitive values are ECS secrets, never plain environment variables', () => {
    for (const container of Object.values(containers(template))) {
      const plain = (container.Environment ?? []).map((e) => e.Name);
      for (const name of SENSITIVE_VARS) expect(plain).not.toContain(name);
    }
    const c = containers(template);
    expect(secretNames(c.core)).toEqual(
      expect.arrayContaining([
        'SECRET_KEY',
        'ANALYSIS_TOKEN_SECRET',
        'STRIPE_SECRET_KEY',
        'STRIPE_PRICE_ID',
        'STRIPE_WEBHOOK_SECRET',
        'DB_USERNAME',
        'DB_PASSWORD',
      ]),
    );
    expect(secretNames(c.analysis)).toEqual(
      expect.arrayContaining(['ANTHROPIC_API_KEY', 'ANALYSIS_TOKEN_SECRET', 'DB_USERNAME', 'DB_PASSWORD']),
    );
    // Least privilege by container: core never sees the Anthropic key.
    expect(secretNames(c.core)).not.toContain('ANTHROPIC_API_KEY');
  });
});

describe('services', () => {
  const template = synth();
  const c = containers(template);

  test('core runs gunicorn run:app in http mode, pointed at the Service Connect name', () => {
    const env = envOf(c.core);
    expect(env.ANALYSIS_MODE).toBe('http');
    expect(env.ANALYSIS_SERVICE_URL).toBe('http://analysis:8000');
    expect(c.core.EntryPoint).toEqual(['sh', '-c']);
    expect(c.core.Command![0]).toContain('export DATABASE_URL="postgresql://');
    expect(c.core.Command![0]).toMatch(/\/core\?sslmode=require"; exec gunicorn run:app /);
  });

  test('analysis gets ANALYSIS_DATABASE_URL and RAG_DATABASE_URL on the analysis database', () => {
    const cmd = c.analysis.Command![0];
    expect(cmd).toMatch(/export ANALYSIS_DATABASE_URL="postgresql:\/\/[^"]+\/analysis\?sslmode=require"/);
    expect(cmd).toMatch(/export RAG_DATABASE_URL="postgresql:\/\/[^"]+\/analysis\?sslmode=require"/);
    expect(cmd).toContain('exec gunicorn analysis_service.wsgi:app');
  });

  test('images come from the ECR repos at the given tag', () => {
    const image = JSON.stringify(c.core.Image);
    expect(image).toContain('fdc/core:abc123');
    expect(JSON.stringify(c.analysis.Image)).toContain('fdc/analysis:abc123');
  });

  test('analysis task has at least 1 GB of memory', () => {
    const td = taskDefFor(template, 'analysis');
    expect(Number(td.Properties.Memory)).toBeGreaterThanOrEqual(1024);
    expect(td.Properties.RequiresCompatibilities).toEqual(['FARGATE']);
  });

  test('container health checks hit /healthz', () => {
    for (const name of ['core', 'analysis']) {
      expect(c[name].HealthCheck!.Command.join(' ')).toContain('http://localhost:8000/healthz');
    }
  });

  test('analysis is published over Service Connect with long request timeouts', () => {
    template.hasResourceProperties('AWS::ECS::Service', {
      LoadBalancers: Match.absent(),
      ServiceConnectConfiguration: {
        Enabled: true,
        Services: [
          {
            PortName: 'http',
            ClientAliases: [{ DnsName: 'analysis', Port: 8000 }],
            Timeout: { PerRequestTimeoutSeconds: 120, IdleTimeoutSeconds: 120 },
          },
        ],
      },
    });
    template.resourceCountIs('AWS::ServiceDiscovery::HttpNamespace', 1);
  });

  test('core is a Service Connect client behind the ALB', () => {
    template.hasResourceProperties('AWS::ECS::Service', {
      LoadBalancers: [Match.objectLike({ ContainerName: 'core', ContainerPort: 8000 })],
      ServiceConnectConfiguration: Match.objectLike({ Enabled: true, Services: Match.absent() }),
    });
  });

  test('deployments roll back on failure', () => {
    template.allResourcesProperties('AWS::ECS::Service', {
      DeploymentConfiguration: Match.objectLike({
        DeploymentCircuitBreaker: { Enable: true, Rollback: true },
      }),
    });
  });
});

describe('load balancer', () => {
  test('internet-facing, health-checking /healthz', () => {
    const template = synth();
    template.hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      Scheme: 'internet-facing',
      LoadBalancerAttributes: Match.arrayWith([{ Key: 'idle_timeout.timeout_seconds', Value: '120' }]),
    });
    template.hasResourceProperties('AWS::ElasticLoadBalancingV2::TargetGroup', {
      Port: 8000,
      TargetType: 'ip',
      HealthCheckPath: '/healthz',
      Matcher: { HttpCode: '200' },
    });
  });

  test('without a certificate: a plain HTTP listener and no DNS records', () => {
    const template = synth();
    template.resourceCountIs('AWS::ElasticLoadBalancingV2::Listener', 1);
    template.hasResourceProperties('AWS::ElasticLoadBalancingV2::Listener', { Port: 80, Protocol: 'HTTP' });
    template.resourceCountIs('AWS::CertificateManager::Certificate', 0);
    template.resourceCountIs('AWS::Route53::RecordSet', 0);
  });

  test('with a domain and hosted zone: DNS-validated cert, HTTPS, HTTP redirect, alias record', () => {
    const template = synth({
      domainName: 'draft.example.com',
      hostedZoneId: 'Z123456ABCDEFG',
      hostedZoneName: 'example.com',
    });
    template.hasResourceProperties('AWS::CertificateManager::Certificate', {
      DomainName: 'draft.example.com',
      ValidationMethod: 'DNS',
    });
    template.hasResourceProperties('AWS::ElasticLoadBalancingV2::Listener', {
      Port: 443,
      Protocol: 'HTTPS',
    });
    template.hasResourceProperties('AWS::ElasticLoadBalancingV2::Listener', {
      Port: 80,
      DefaultActions: [Match.objectLike({ Type: 'redirect', RedirectConfig: Match.objectLike({ Port: '443' }) })],
    });
    template.hasResourceProperties('AWS::Route53::RecordSet', {
      Name: 'draft.example.com.',
      Type: 'A',
    });
  });

  test('adding a certificate updates the port-80 listener and target group in place', () => {
    const port80Id = (t: Template) =>
      Object.entries(t.findResources('AWS::ElasticLoadBalancingV2::Listener', { Properties: { Port: 80 } })).map(
        ([id]) => id,
      );
    const targetGroupIds = (t: Template) => Object.keys(t.findResources('AWS::ElasticLoadBalancingV2::TargetGroup'));

    const before = synth();
    const after = synth({
      domainName: 'draft.example.com',
      certificateArn: 'arn:aws:acm:us-east-1:111111111111:certificate/abc',
    });
    expect(port80Id(before)).toHaveLength(1);
    expect(port80Id(after)).toEqual(port80Id(before));
    expect(targetGroupIds(after)).toEqual(targetGroupIds(before));
  });

  test('the HTTPS redirect and DNS record wait for the HTTPS listener', () => {
    const template = synth({
      domainName: 'draft.example.com',
      hostedZoneId: 'Z123456ABCDEFG',
      hostedZoneName: 'example.com',
    });
    const [httpsId] = Object.keys(
      template.findResources('AWS::ElasticLoadBalancingV2::Listener', { Properties: { Port: 443 } }),
    );
    const [http] = Object.values(
      template.findResources('AWS::ElasticLoadBalancingV2::Listener', { Properties: { Port: 80 } }),
    );
    const [alias] = Object.values(template.findResources('AWS::Route53::RecordSet', { Properties: { Type: 'A' } }));
    expect(httpsId).toBeDefined();
    expect(http.DependsOn).toContain(httpsId);
    expect(alias.DependsOn).toContain(httpsId);
    // ...and the HTTPS listener itself waits for the validated certificate.
    const [certId] = Object.keys(template.findResources('AWS::CertificateManager::Certificate'));
    const https = template.toJSON().Resources[httpsId];
    expect(JSON.stringify(https.Properties.Certificates)).toContain(certId);
  });

  test('deregistration delay lets in-flight /analyze requests finish', () => {
    synth().hasResourceProperties('AWS::ElasticLoadBalancingV2::TargetGroup', {
      TargetGroupAttributes: Match.arrayWith([{ Key: 'deregistration_delay.timeout_seconds', Value: '120' }]),
    });
  });

  test('with an existing certificate ARN: no new certificate', () => {
    const template = synth({
      domainName: 'draft.example.com',
      certificateArn: 'arn:aws:acm:us-east-1:111111111111:certificate/abc',
    });
    template.resourceCountIs('AWS::CertificateManager::Certificate', 0);
    template.hasResourceProperties('AWS::ElasticLoadBalancingV2::Listener', {
      Port: 443,
      Certificates: [{ CertificateArn: 'arn:aws:acm:us-east-1:111111111111:certificate/abc' }],
    });
  });

  test('rejects a half-specified hosted zone', () => {
    expect(() => synth({ domainName: 'draft.example.com', hostedZoneId: 'Z123' })).toThrow(/together/);
  });
});

describe('migration task', () => {
  test('reads Render URLs from a secret and restores into both databases', () => {
    const c = containers(synth());
    expect(secretNames(c.migrate)).toEqual(
      expect.arrayContaining(['RENDER_DATABASE_URL', 'RENDER_ANALYSIS_DATABASE_URL', 'DB_PASSWORD']),
    );
    const script = c.migrate.Command![0];
    expect(script).toContain('pg_dump --format=custom');
    expect(script).toContain('pg_restore --clean --if-exists');
    expect(script).toMatch(/restore "\$RENDER_DATABASE_URL" core/);
    expect(script).toMatch(/restore "\$RENDER_ANALYSIS_DATABASE_URL" analysis/);
  });
});

describe('RAG index task', () => {
  const template = synth();
  const c = containers(template);

  test('downloads the data, then indexes into the analysis database', () => {
    const cmd = c['rag-index'].Command![0];
    expect(cmd).toMatch(/^export RAG_DATABASE_URL="postgresql:\/\/[^"]+\/analysis\?sslmode=require"; /);
    // Both steps must run: a bare `exec a && b` would never reach b.
    expect(cmd).toContain(`exec sh -c '${RAG_INDEX_COMMAND}'`);
    expect(RAG_INDEX_COMMAND).toBe('python -m ml.download && python -m ml.rag.index');
    expect(JSON.stringify(c['rag-index'].Image)).toContain('fdc/rag-indexer:abc123');
    expect(secretNames(c['rag-index'])).toEqual(expect.arrayContaining(['DB_USERNAME', 'DB_PASSWORD']));
  });

  test('is a task definition only, not a service', () => {
    const td = taskDefFor(template, 'rag-index');
    expect(Number(td.Properties.Memory)).toBeGreaterThanOrEqual(2048);
    template.resourceCountIs('AWS::ECS::Service', 2);
  });
});

describe('withDatabaseUrls', () => {
  test('exports one URL per variable and execs the command', () => {
    expect(withDatabaseUrls({ A_URL: 'one', B_URL: 'two' }, 'gunicorn x')).toBe(
      'export A_URL="postgresql://$DB_USERNAME:$DB_PASSWORD@$DB_HOST:$DB_PORT/one?sslmode=require"; ' +
        'export B_URL="postgresql://$DB_USERNAME:$DB_PASSWORD@$DB_HOST:$DB_PORT/two?sslmode=require"; ' +
        'exec gunicorn x',
    );
  });
});
