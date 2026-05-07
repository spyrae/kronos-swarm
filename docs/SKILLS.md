# Kronos Agent OS (KAOS) — Skills

Skills are workspace-local procedures that let an agent load reusable behavior
on demand without putting every instruction into the system prompt.

## Mental Model

KAOS uses progressive disclosure:

| Level | Content | When Loaded |
|-------|---------|-------------|
| L1 Catalog | skill name + short description | included in the system prompt |
| L2 Protocol | full `SKILL.md` | loaded with `load_skill(name)` |
| L3 References | supporting files | loaded with `load_skill_reference(name, ref)` |

This keeps the base prompt small while still allowing deep task-specific
behavior.

## Location

Skills live inside a local agent workspace:

```text
workspaces/<agent>/self/skills/
└── research-brief/
    ├── SKILL.md
    └── references/
        └── SOURCES.md
```

Live `workspaces/<agent>/` directories are gitignored because they can contain
private notes, memory references, and operational state. Public examples should
be scrubbed and stored outside a live workspace.

## Minimal Skill

```markdown
---
name: research-brief
description: Create a concise research brief with sources and next actions.
capabilities:
  - web_search
risk: low
---

# Research Brief

## Trigger

Use when the user asks for a short research summary or decision brief.

## Protocol

1. Clarify the question in one sentence.
2. Gather current sources when freshness matters.
3. Summarize findings, tradeoffs, and confidence.
4. End with concrete next actions.

## Output

- Summary
- Evidence
- Risks
- Next actions
```

## Loading Flow

1. `SkillStore` scans `workspaces/<agent>/self/skills/`.
2. The system prompt receives the L1 catalog.
3. The agent calls `load_skill("research-brief")` when the task matches.
4. The agent may call `load_skill_reference(...)` for supporting files.
5. Skill changes are local until the user intentionally publishes a scrubbed example.

## Recommended Structure

```text
SKILL.md
references/
  EXAMPLE.md
fixtures/
  input.md
  expected.md
```

Use front matter for metadata and keep the body readable without hidden
side effects:

| Field | Purpose |
|-------|---------|
| `name` | stable slug used by loader/tools |
| `description` | short L1 catalog text |
| `capabilities` | required tool/capability categories |
| `risk` | `low`, `medium`, or `high` operator hint |

The current runtime treats skill files as local procedures. Capability metadata
is documentation and review signal; runtime gates still live in config/tool
checks.

## Install, Load, Disable

Current local flow:

- Install by placing the skill directory under `workspaces/<agent>/self/skills/`.
- Load at runtime through `load_skill(name)`.
- Load supporting files through `load_skill_reference(name, ref)`.
- Disable by removing or moving the skill directory out of the workspace.

Bundled packs live in `templates/skill-packs/`:

```bash
kaos skills packs
kaos skills show-pack research
kaos skills install-pack research --agent personal-operator --dry-run
```

Pack installation copies `skills/*` into `workspaces/<agent>/self/skills/`.
Existing skills are skipped unless `--force` is provided.

Portable skills can also be imported from a direct `SKILL.md` URL or GitHub
source:

```bash
kaos skills import https://example.com/path/SKILL.md --agent personal-operator
kaos skills import github:org/repo/research-brief --agent personal-operator
kaos skills export research-brief --agent personal-operator --output /tmp/SKILL.md
```

External imports are always written as `status: draft` with source metadata
(`imported_from`, `source_url`, `imported_at`) and `review_required: true`.
Review with `load_skill(...)`, then approve explicitly with `approve_skill(...)`
if the procedure is safe and useful.

## Skills Vs MCP Tools

| Extension | Best For | Executes Code? |
|-----------|----------|----------------|
| Skill | reusable instructions, procedures, references, output formats | no |
| MCP tool | external capability such as search, files, APIs, browser, data stores | yes |
| Built-in tool | first-party KAOS runtime capability | yes |

Use a skill when the agent needs to know how to do something. Use a tool when
the agent needs to actually touch a system.

## Demo-Safe Example

The `research-brief` example above is demo-safe because it can run as pure
procedure text. It may recommend gathering sources, but it should only call
search/MCP tools when those tools are configured and approved by the runtime
capability policy.

## Safety Rules

- Do not hide risky tool behavior inside skill prose.
- Document required tools and capability gates in the skill.
- Do not commit private references, customer data, credentials, or live workspace notes.
- Keep destructive actions behind explicit user confirmation and runtime capability checks.
- Keep high-risk skills opt-in and explain which env vars enable the risky capability.

## Recommended Public Examples

Good public skills:

- `research-brief` — source-backed research summary
- `release-notes` — convert commits/issues into release notes
- `incident-review` — summarize logs and follow-up tasks
- `meeting-brief` — prepare agenda and decisions

Avoid public examples that require private accounts, real financial data,
private Telegram groups, production servers, or personal memory files.
