# MCP and Tool Gateway

KAOS treats tools as capabilities, not as an unbounded execution surface.

## Tool Types

| Type | Examples | Default |
|------|----------|---------|
| Static MCP tools | fetch, search, filesystem when workspace exists | allowed when configured |
| Built-in tools | session search, skill tools, browser tools | allowed when dependency exists |
| Dynamic tools | generated local Python tools | disabled |
| Dynamic MCP servers | persisted runtime MCP registration | disabled |
| Server ops tools | SSH/systemd/docker diagnostics | disabled |

## Capability Gates

```bash
ENABLE_DYNAMIC_TOOLS=false
REQUIRE_DYNAMIC_TOOL_SANDBOX=true
ENABLE_MCP_GATEWAY_MANAGEMENT=false
ENABLE_DYNAMIC_MCP_SERVERS=false
ENABLE_SERVER_OPS=false
```

Keep these disabled for public demos and untrusted environments.

## Static MCP

Static MCP servers are configured in code and loaded by the runtime manager.
Providers that need API keys should be optional: missing keys should skip the
server instead of crashing the runtime.

The filesystem MCP server is only added when the configured workspace path
exists.

Discovery path:

1. `kronos/tools/manager.py` builds configured static MCP connections.
2. Available tools are handed to `KronosAgent`.
3. The ReAct engine binds the tools to the model.
4. Tool calls are executed through the runtime loop and summarized in logs/CLI.

Safe local examples:

- read/search tools scoped to the workspace
- web search tools with explicit query text
- session search over local KAOS history

High-risk examples:

- shell/filesystem writes outside the workspace
- adding or reloading MCP servers at runtime
- server ops, SSH, Docker, or systemd actions
- tools that can spend money, send messages, or mutate external systems

## Dynamic MCP

Runtime server management is powerful and should be treated as a local admin
feature. It is unavailable unless:

```bash
ENABLE_MCP_GATEWAY_MANAGEMENT=true
```

Persisted dynamic servers are ignored unless:

```bash
ENABLE_DYNAMIC_MCP_SERVERS=true
```

## Dynamic Tools

Dynamic tool creation is disabled unless:

```bash
ENABLE_DYNAMIC_TOOLS=true
```

When dynamic tools are enabled, the public-safe expectation is sandboxed
execution:

```bash
REQUIRE_DYNAMIC_TOOL_SANDBOX=true
```

Build the local sandbox image before enabling dynamic tools:

```bash
scripts/build-sandbox.sh
ENABLE_DYNAMIC_TOOLS=true kaos doctor
```

If Docker or the sandbox image is unavailable, dynamic execution fails closed.

## Server Ops

Server ops require explicit opt-in and a private registry:

```bash
ENABLE_SERVER_OPS=true
SERVER_REGISTRY_PATH=/path/to/servers.yaml
```

Use `servers.example.yaml` as the public shape. Do not commit `servers.yaml`.

## Approvals, Audit, And Errors

The current public posture is capability-gated:

- blocked capabilities should name the env var needed for opt-in
- dynamic/server operations should fail closed when the gate is disabled
- CLI chat prints compact tool events with secret-like args redacted
- dashboard audit views should show tool/runtime events without exposing secrets
- missing optional API keys should skip the integration rather than crash startup

For user-visible commands, prefer a short refusal plus the exact gate to change
over silent no-ops.
