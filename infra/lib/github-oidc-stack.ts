import { CfnOutput, Duration, Stack, StackProps } from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';
import { ANALYSIS_REPO_NAME, CORE_REPO_NAME } from './config';

export interface GithubOidcStackProps extends StackProps {
  /** `owner/name` of the GitHub repository allowed to assume the role. */
  readonly githubRepo: string;
  /** Branch whose workflow runs may deploy. */
  readonly branch?: string;
  /**
   * ARN of an existing token.actions.githubusercontent.com provider. An
   * account can hold only one, so pass it if another project already made it.
   */
  readonly existingProviderArn?: string;
}

/**
 * The role GitHub Actions assumes through OIDC: no long-lived AWS keys in the
 * repo. Deployed once by hand during bootstrap (see infra/README.md); its
 * ARN goes into the AWS_DEPLOY_ROLE_ARN repository variable.
 *
 * The role can only push to the two ECR repositories and assume the CDK
 * bootstrap roles. CloudFormation changes run as the bootstrap deploy role,
 * not as this one.
 */
export class GithubOidcStack extends Stack {
  readonly role: iam.Role;

  constructor(scope: Construct, id: string, props: GithubOidcStackProps) {
    super(scope, id, props);

    const branch = props.branch ?? 'main';
    const issuer = 'token.actions.githubusercontent.com';

    const providerArn = props.existingProviderArn
      ? props.existingProviderArn
      : new iam.OidcProviderNative(this, 'GithubOidcProvider', {
          url: `https://${issuer}`,
          clientIds: ['sts.amazonaws.com'],
        }).oidcProviderArn;

    this.role = new iam.Role(this, 'DeployRole', {
      description: `GitHub Actions deploys from ${props.githubRepo}@${branch}`,
      maxSessionDuration: Duration.hours(1),
      assumedBy: new iam.WebIdentityPrincipal(providerArn, {
        StringEquals: {
          [`${issuer}:aud`]: 'sts.amazonaws.com',
          // Only workflow runs on the deploy branch; not PRs, not forks.
          [`${issuer}:sub`]: `repo:${props.githubRepo}:ref:refs/heads/${branch}`,
        },
      }),
    });

    // `cdk deploy` works by assuming the roles `cdk bootstrap` created.
    this.role.addToPolicy(
      new iam.PolicyStatement({
        actions: ['sts:AssumeRole'],
        resources: [`arn:${this.partition}:iam::${this.account}:role/cdk-*`],
      }),
    );

    this.role.addToPolicy(
      new iam.PolicyStatement({ actions: ['ecr:GetAuthorizationToken'], resources: ['*'] }),
    );
    this.role.addToPolicy(
      new iam.PolicyStatement({
        actions: [
          'ecr:BatchCheckLayerAvailability',
          'ecr:BatchGetImage',
          'ecr:CompleteLayerUpload',
          'ecr:DescribeImages',
          'ecr:GetDownloadUrlForLayer',
          'ecr:InitiateLayerUpload',
          'ecr:PutImage',
          'ecr:UploadLayerPart',
        ],
        resources: [CORE_REPO_NAME, ANALYSIS_REPO_NAME].map(
          (name) => `arn:${this.partition}:ecr:${this.region}:${this.account}:repository/${name}`,
        ),
      }),
    );

    new CfnOutput(this, 'DeployRoleArn', {
      value: this.role.roleArn,
      description: 'Set as the AWS_DEPLOY_ROLE_ARN repository variable in GitHub',
    });
  }
}
