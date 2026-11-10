/**
 * Names shared between stacks, the GitHub workflow and the README.
 * Change them here, not in the stacks.
 */

/** ECR repository names. The deploy workflow pushes to these. */
export const CORE_REPO_NAME = 'fdc/core';
export const ANALYSIS_REPO_NAME = 'fdc/analysis';

/** Both images listen here (see the Dockerfiles' gunicorn --bind). */
export const CONTAINER_PORT = 8000;

/** Service Connect name the core app uses to reach the analysis service. */
export const ANALYSIS_DNS_NAME = 'analysis';

/** Database names on the shared RDS instance. */
export const CORE_DB_NAME = 'core';
export const ANALYSIS_DB_NAME = 'analysis';

/**
 * Container commands. ECS can't build a URL out of secret fields, so each app
 * container starts under `sh -c`, assembles its database URLs from the
 * injected DB_* variables, then `exec`s the same gunicorn command its
 * Dockerfile's CMD runs. Keep these in step with the Dockerfiles.
 */
export const CORE_GUNICORN =
  'gunicorn run:app --bind 0.0.0.0:8000 --workers 2 --timeout 120';
export const ANALYSIS_GUNICORN =
  'gunicorn analysis_service.wsgi:app --bind 0.0.0.0:8000 --workers 2 --timeout 120';

/** Secret names (fixed so the README can give copy-pasteable commands). */
export const SECRET_NAMES = {
  anthropicApiKey: 'fdc/anthropic-api-key',
  secretKey: 'fdc/secret-key',
  analysisTokenSecret: 'fdc/analysis-token-secret',
  stripe: 'fdc/stripe',
  renderMigration: 'fdc/render-migration',
} as const;

/** Placeholder written into secrets a human must fill in after the first deploy. */
export const PLACEHOLDER = 'REPLACE_ME';
