import { App } from 'aws-cdk-lib';
import { Match, Template } from 'aws-cdk-lib/assertions';
import { EcrStack } from '../lib/ecr-stack';
import { GithubOidcStack } from '../lib/github-oidc-stack';

describe('EcrStack', () => {
  const template = Template.fromStack(new EcrStack(new App(), 'TestEcr'));

  test.each(['fdc/core', 'fdc/analysis'])('%s repo scans on push and is retained', (name) => {
    template.hasResource('AWS::ECR::Repository', {
      Properties: Match.objectLike({
        RepositoryName: name,
        ImageScanningConfiguration: { ScanOnPush: true },
        LifecyclePolicy: Match.anyValue(),
      }),
      DeletionPolicy: 'Retain',
    });
  });
});

describe('GithubOidcStack', () => {
  const synth = (existingProviderArn?: string) =>
    Template.fromStack(
      new GithubOidcStack(new App(), 'TestOidc', {
        githubRepo: 'owner/repo',
        existingProviderArn,
      }),
    );

  test('creates the GitHub OIDC provider when none is given', () => {
    synth().hasResourceProperties('AWS::IAM::OIDCProvider', {
      Url: 'https://token.actions.githubusercontent.com',
      ClientIdList: ['sts.amazonaws.com'],
    });
  });

  test('reuses an existing provider', () => {
    const template = synth('arn:aws:iam::111111111111:oidc-provider/token.actions.githubusercontent.com');
    template.resourceCountIs('AWS::IAM::OIDCProvider', 0);
    template.hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [
          Match.objectLike({
            Principal: {
              Federated: 'arn:aws:iam::111111111111:oidc-provider/token.actions.githubusercontent.com',
            },
          }),
        ],
      },
    });
  });

  test('only main-branch runs of the repo can assume the deploy role', () => {
    synth().hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [
          Match.objectLike({
            Action: 'sts:AssumeRoleWithWebIdentity',
            Condition: {
              StringEquals: {
                'token.actions.githubusercontent.com:aud': 'sts.amazonaws.com',
                'token.actions.githubusercontent.com:sub': 'repo:owner/repo:ref:refs/heads/main',
              },
            },
          }),
        ],
      },
    });
  });

  test('can assume the CDK bootstrap roles and push to the two repos, nothing broader', () => {
    const template = synth();
    const policies = Object.values(template.findResources('AWS::IAM::Policy'));
    const statements = policies.flatMap((p) => p.Properties.PolicyDocument.Statement);
    const actions = statements.flatMap((s: { Action: string | string[] }) => [s.Action].flat());
    expect(actions).toContain('sts:AssumeRole');
    expect(actions).toContain('ecr:PutImage');
    expect(actions.some((a: string) => a === '*' || a.endsWith(':*'))).toBe(false);
    const json = JSON.stringify(statements);
    expect(json).toContain(':role/cdk-*');
    expect(json).toContain('repository/fdc/core');
    expect(json).toContain('repository/fdc/analysis');
  });
});
