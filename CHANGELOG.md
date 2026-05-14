# Changelog

All notable changes to Kronos Agent OS are documented here.

## [0.1.1] - 2026-04-28

### Added

- Telegram topic routing for group chats with multiple threads.
- Telegram session sidecar preservation across restarts.
- Codex OAuth integration for orchestrator routing.
- Expanded agent runtime capabilities.
- PyPI badges and `pip install kronos-agent-os` quickstart in README.

### Fixed

- Telegram model identity now correctly replies under multi-agent setups.
- Competitor monitor startup restored.
- Deploy health check retry logic.
- Peer messages in owner-only topics correctly ignored.

### Changed

- Hardened Codex and MCP runtime config.
- Trimmed deploy sync artifacts for faster deployments.

## [0.1.0] - 2026-04-27

Initial public release.

### Added

- Custom ReAct-style engine.
- Main agent pipeline: validate, memory, route, store, compact.
- Pydantic settings configuration.
- Session memory, FTS5 recall, Mem0 vector memory, and knowledge graph.
- Workspace-local skills and references.
- MCP and custom tool gateway.
- Scheduled jobs for digests, monitoring, analytics, and maintenance.
- Dashboard/API for runtime inspection.
- Optional swarm coordination with SQLite claim arbitration.
- Telethon userbot bridge, Discord bridge, webhook server.
- Public CLI: `kaos doctor`, `kaos init`, `kaos demo`, `kaos chat`, `kaos connect telegram`.
- One-shot chat mode: `kaos chat --prompt` and `--no-memory`.

### Changed

- Reframed the project as Kronos Agent OS (KAOS), not only swarm/council coordination.
- Made `kaos demo` an offline deterministic walkthrough (no Telegram, Docker, or LLM keys required).
- Made live `workspaces/<agent>/` local runtime state; only `workspaces/_template` is public.
- Hardened dashboard defaults: localhost binding and generated password when unset.
- Made Docker quickstart safer with localhost-only port bindings and `.dockerignore`.
- Sanitized public examples, docs, scripts, systemd units, ASO defaults, and dashboard labels.

### Security

- Prompt injection shield, output validation, cost guardrails, and loop detection.
- Dynamic tools, dynamic MCP management, dynamic MCP registry loading, and server ops disabled by default.
- Telegram DMs blocked unless `ALLOWED_USERS` is set or `ALLOW_ALL_USERS=true`.
- Server operations require explicit opt-in plus a private `servers.yaml`.

### Testing

- Regression coverage for capability gates, Docker quickstart, offline demo, CLI parsing, and public workspace surface.
