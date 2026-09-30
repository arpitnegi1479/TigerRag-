# Security and Hardening

## Document inputs

The upload route accepts only configured extensions, rejects path components in filenames, applies the configured maximum size while reading the upload, and maps parser failures to a generic client error. Parser exception details are logged by exception type only and are not returned to clients. Supported MIME types and PDF signatures are checked. Documents are untrusted evidence: grounded prompts explicitly label retrieved text as untrusted, escape angle brackets before inserting it inside `<retrieved_evidence>` tags, and keep agent state inside its own escaped delimiter. This prevents document text from forging a closing prompt tag.

## Request validation and errors

FastAPI/Pydantic validates request bodies and returns 422 for malformed payloads. Upload validation failures use 400; oversized uploads use 413. Evaluation infrastructure failures use 503. Responses avoid returning exception traces or credential values.

## Rate limiting

The application uses an in-memory, per-client sliding-window limiter. Defaults are 120 requests per 60 seconds and can be changed with `RATE_LIMIT_REQUESTS_PER_MINUTE` and `RATE_LIMIT_WINDOW_SECONDS`. A rejected request receives HTTP 429 and a `Retry-After` header. The limiter uses the socket peer address and does not trust forwarded-IP headers. Its state is per application process; deployments with multiple workers or replicas need an upstream shared limiter for a global quota.

## Credentials and operations

Gemini, Postgres, and Neo4j credentials are loaded from environment configuration; the app and Compose configuration contain no database password defaults. Set `POSTGRES_PASSWORD` and `NEO4J_PASSWORD` in the ignored local `.env` before starting Compose. Missing Gemini credentials cause provider generation to fail with a sanitized error; retrieval paths can use explicitly reported fallback behavior. Avoid logging API keys, raw provider URLs, or untrusted document instructions. Evaluation quality is marked degraded whenever a tracked Gemini-dependent step fails or falls back.

## Verification coverage

`tests/test_security.py` covers disallowed extensions, oversized upload rejection, an uploaded prompt-injection document flowing through retrieval into the untrusted evidence delimiter, malformed API payloads, malformed PDF parsing, missing Gemini credentials, and rate limiting. These checks are regression coverage, not a substitute for deployment-level proxy, TLS, authentication, or abuse testing.
