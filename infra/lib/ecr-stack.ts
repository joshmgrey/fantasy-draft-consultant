import { RemovalPolicy, Stack, StackProps } from 'aws-cdk-lib';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import { Construct } from 'constructs';
import { ANALYSIS_REPO_NAME, CORE_REPO_NAME } from './config';

/**
 * The two image repositories, in their own stack so CI can create them before
 * it pushes the first images and before the app stack (whose task definitions
 * point at those images) is deployed.
 */
export class EcrStack extends Stack {
  readonly coreRepo: ecr.Repository;
  readonly analysisRepo: ecr.Repository;

  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const repo = (logicalId: string, repositoryName: string) =>
      new ecr.Repository(this, logicalId, {
        repositoryName,
        imageScanOnPush: true,
        // CI pushes `<git sha>` and moves `latest`. Rolling back is
        // `cdk deploy FdcApp -c imageTag=<older sha>`, so keep a history.
        imageTagMutability: ecr.TagMutability.MUTABLE,
        lifecycleRules: [{ description: 'Keep the 30 most recent images', maxImageCount: 30 }],
        // Deleting the stack must not take the images (and rollback) with it.
        removalPolicy: RemovalPolicy.RETAIN,
      });

    this.coreRepo = repo('CoreRepo', CORE_REPO_NAME);
    this.analysisRepo = repo('AnalysisRepo', ANALYSIS_REPO_NAME);
  }
}
