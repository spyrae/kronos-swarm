# Security Policy

Kronos Agent OS (KAOS) is a self-hosted agent runtime. It can connect to LLMs, local memory, MCP tools, scheduled jobs, browser tools, Telegram/Discord, and optional infrastructure tooling. Treat it like a local automation system with access to whatever credentials and tools you enable.

## Supported Versions

Security fixes target the latest public release and `main` until the project has a broader release policy.

## Reporting

Do not open public issues containing secrets, exploit details against real systems, or private infrastructure data. Report security-sensitive issues privately to the maintainer first.

Include:

- Affected version or commit.
- Configuration used, especially enabled capability gates.
- Minimal reproduction steps.
- Impact and expected behavior.

## Public-Safe Defaults

Fresh installs are conservative:

- `ENABLE_DYNAMIC_TOOLS=false`
- `REQUIRE_DYNAMIC_TOOL_SANDBOX=true`
- `ENABLE_MCP_GATEWAY_MANAGEMENT=false`
- `ENABLE_DYNAMIC_MCP_SERVERS=false`
- `ENABLE_SERVER_OPS=false`

These defaults are intentional. Enable risky capabilities only in trusted local environments.

When enabling dynamic tools, build the local sandbox image with `scripts/build-sandbox.sh` first. `kaos doctor` reports a hard failure if `ENABLE_DYNAMIC_TOOLS=true` but Docker or `kronos-sandbox:latest` is unavailable.

## Threat Model

Primary risks:

- Prompt injection through user messages, web pages, email, Telegram, Discord, and external documents.
- Secret leakage from `.env`, MCP server env vars, logs, tool arguments, and model output.
- Unsafe dynamic code or tool execution.
- Unsafe runtime mutation through dynamic MCP server registration.
- Accidental infrastructure actions through server ops tools.
- Scheduled jobs repeating a harmful or expensive action.
- Dashboard exposure beyond localhost without proper auth and network controls.

Security controls currently include input validation, external-content sanitization, output redaction, PII masking for logs/traces/audit previews, loop detection, cost guarding, capability gates, and sandbox-required dynamic tool execution.

See [docs/SECURITY.md](docs/SECURITY.md) for implementation details.

## Responsible Use

KAOS is intended for user-authorized personal, research, productivity, and operations workflows. Do not use it to access systems without permission, bypass security controls, harvest credentials, or run automation against third-party services in ways that violate their rules.
