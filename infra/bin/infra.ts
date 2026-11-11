#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { AppStack } from '../lib/app-stack';
import { EcrStack } from '../lib/ecr-stack';
import { GithubOidcStack } from '../lib/github-oidc-stack';

const app = new cdk.App();

/** A context value (`-c key=value` or cdk.json), with '' treated as unset. */
const ctx = (key: string): string | undefined => {
  const value = app.node.tryGetContext(key);
  return typeof value === 'string' && value.trim() !== '' ? value.trim() : undefined;
};

// Account and region come from the caller's AWS credentials / AWS_REGION.
// Without them (e.g. `cdk synth` on a laptop with no profile) the stacks
// synthesize environment-agnostic, which is all the tests need.
const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION,
};

new GithubOidcStack(app, 'FdcGithubOidc', {
  env,
  githubRepo: ctx('githubRepo') ?? 'joshmgrey/fantasy-draft-consultant',
  existingProviderArn: ctx('githubOidcProviderArn'),
});

new EcrStack(app, 'FdcEcr', { env });

new AppStack(app, 'FdcApp', {
  env,
  imageTag: ctx('imageTag') ?? 'latest',
  domainName: ctx('domainName'),
  hostedZoneId: ctx('hostedZoneId'),
  hostedZoneName: ctx('hostedZoneName'),
  certificateArn: ctx('certificateArn'),
  migrateImage: ctx('migrateImage'),
});

cdk.Tags.of(app).add('project', 'fantasy-draft-consultant');
