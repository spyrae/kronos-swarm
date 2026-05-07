# Kronos Agent OS (KAOS) — Cron Jobs

All 11 cron jobs are registered in `kronos/cron/setup.py` and run by the built-in async scheduler (`kronos/cron/scheduler.py`). No external dependencies (no APScheduler). Runs inside the main event loop alongside bridges and dashboard.

## Schedule Overview

| # | Name | Schedule | Type | Module |
|---|------|----------|------|--------|
| 1 | heartbeat | Every 30 min | Periodic | `heartbeat.py` |
| 2 | news-monitor | Daily 00:30 UTC (08:30 UTC+8) | Daily | `news_monitor.py` |
| 3 | group-digest | Daily 01:00 UTC (09:00 UTC+8) | Daily | `group_digest.py` |
| 4 | email-expenses | Daily 00:00 UTC (08:00 UTC+8) | Daily | `email_expenses.py` |
| 5 | sleep-compute | Daily 03:00 UTC (11:00 UTC+8) | Daily | `sleep_compute.py` |
| 6 | self-improve | Daily 22:00 UTC (06:00 UTC+8) | Daily | `self_improve.py` |
| 7 | expense-digest | Weekly Sun 02:00 UTC (10:00 UTC+8) | Weekly | `expense_digest.py` |
| 8 | people-scout | Weekly Sun 02:00 UTC (10:00 UTC+8) | Weekly | `people_scout.py` |
| 9 | skill-improve | Weekly Sun 20:00 UTC (04:00 UTC+8) | Weekly | `skill_improve.py` |
| 10 | user-model | Weekly Wed 20:00 UTC (04:00 UTC+8) | Weekly | `user_model.py` |
| 11 | market-review | Weekly Fri 10:00 UTC (18:00 UTC+8) | Weekly | `market_review.py` |

## Job Details

### 1. heartbeat
**Schedule:** Every 30 minutes
**Module:** `kronos/cron/heartbeat.py`

Reads `workspace/HEARTBEAT.md` tasks + queries Notion DB for current tasks. Sends to DeepSeek (lite) for analysis. Only notifies if something actionable (overdue deadlines, important reminders). Silent if everything is fine ("heartbeat: ok").

**Dependencies:** HEARTBEAT.md, Notion API (optional)
**Notification:** Webhook → Telegram DM

### 2. news-monitor
**Schedule:** Daily 00:30 UTC
**Module:** `kronos/cron/news_monitor.py`

Daily news digest pipeline:
1. Load watchlist from `workspace/skills/news-monitor/references/WATCHLIST.md`
2. Parse Reddit subreddits and Twitter accounts into search queries
3. Brave Search for each topic (freshness=past day, up to 10 topics)
4. LLM synthesis (DeepSeek lite) → structured HTML digest
5. Send to Telegram via Bot API (NEWS_TOPIC_ID)

**Dependencies:** BRAVE_API_KEY, WATCHLIST.md
**Notification:** Bot API → Telegram News topic

### 3. group-digest
**Schedule:** Daily 01:00 UTC
**Module:** `kronos/cron/group_digest.py`

Daily Telegram group digest:
1. Load groups from `workspace/skills/group-digest/references/GROUPS.md`
2. For each group: fetch last 24h messages via Telethon (max 200 per group)
3. Filter significant messages by engagement (reactions >= 3 or views >= 200)
4. Score and rank: `reactions * 10 + views / 100`
5. LLM synthesis (DeepSeek lite) → HTML digest with insights
6. Send to Telegram via Bot API (DIGEST_TOPIC_ID)

**Dependencies:** Telethon client (shared), GROUPS.md
**Notification:** Bot API → Telegram Digest topic

### 4. email-expenses
**Schedule:** Daily 00:00 UTC
**Module:** `kronos/cron/email_expenses.py`

Auto-extract expenses from Gmail receipts:
1. Search Gmail for receipt/invoice emails (requires Google Workspace MCP)
2. LLM extracts expense data (description, amount, currency, category, date)
3. Create entries in Notion Expenses DB

**Dependencies:** NOTION_API_KEY, Google OAuth (for Gmail)
**Status:** Partially implemented — Gmail search requires MCP integration, currently a stub
**Notification:** Webhook → Telegram DM

### 5. sleep-compute
**Schedule:** Daily 03:00 UTC (L4 Memory)
**Module:** `kronos/cron/sleep_compute.py`

Nightly memory consolidation:
1. Get recent facts from FTS5 (last 7 days)
2. LLM extracts entities and relationships → Knowledge Graph (DeepSeek lite)
3. Build/update entity relations in SQLite
4. Generate 1-3 actionable insights from graph patterns
5. Clean up stale facts (>90 days) from FTS5

**Dependencies:** FTS5 database, Knowledge Graph database
**Notification:** Webhook → Telegram DM (entities added, relations, insights)

### 6. self-improve
**Schedule:** Daily 22:00 UTC
**Module:** `kronos/cron/self_improve.py`

Daily agent self-improvement:
1. Read last 24h from `audit.jsonl` (up to 20 entries)
2. Load previous improvements from `workspace/memory/self-improve/`
3. LLM analyzes sessions → proposes ONE concrete improvement (DeepSeek lite)
4. Save as dated learning record (YYYY-MM-DD.md)
5. Skip if "no improvements needed"

**Dependencies:** audit.jsonl
**Notification:** Webhook → Telegram DM

### 7. expense-digest
**Schedule:** Weekly Sunday 02:00 UTC
**Module:** `kronos/cron/expense_digest.py`

Weekly expense report:
1. Query Notion Expenses DB for last 7 days
2. LLM analysis (DeepSeek lite): totals, by category, top 3 expenses, trend, recommendation
3. Send HTML report to Telegram

**Dependencies:** NOTION_API_KEY, user-configured expenses database
**Notification:** Bot API → Telegram Finance topic

### 8. people-scout
**Schedule:** Weekly Sunday 02:00 UTC
**Module:** `kronos/cron/people_scout.py`

LinkedIn profile discovery:
1. Rotate focus weekly: US founders → EU founders → AI engineers → Indie hackers
2. LLM generates profiles based on criteria (Sonnet standard)
3. Deduplicate against `SEEN.md`
4. Extract LinkedIn URLs → update SEEN.md
5. Send HTML report to Telegram

**Dependencies:** CRITERIA.md, SEEN.md
**Notification:** Bot API → Telegram Scout topic

### 9. skill-improve
**Schedule:** Weekly Sunday 20:00 UTC
**Module:** `kronos/cron/skill_improve.py`

Auto-improvement of skill files:
1. Read last 7 days from `audit.jsonl`
2. Match interactions to skills by keywords (expense-tracker, investment-analysis, etc.)
3. For skills with >= 3 interactions: LLM proposes minimal improvement (DeepSeek lite)
4. Backup current SKILL.md → `.versions/SKILL.vN.md`
5. Write updated SKILL.md

**Dependencies:** audit.jsonl, skill SKILL.md files
**Notification:** Webhook → Telegram DM

### 10. user-model
**Schedule:** Weekly Wednesday 20:00 UTC
**Module:** `kronos/cron/user_model.py`

Dialectical user modeling:
1. **Quantitative**: Pure Python analytics on audit.jsonl (peak hours, avg message length, tier distribution, response time)
2. **Qualitative**: LLM analyzes last 30 conversations plus decision/preference snippets from session search
3. **Dialectical**: Compare against previous model → validate/update/add hypotheses
4. **Passive quality signals**: correction requests, slow responses, tool-heavy sessions, errors, and cost patterns without requiring likes/reactions
5. Categories: Beliefs, Motivations, Decision Patterns, Tensions, Evolution
6. Each belief has numeric confidence: 0.0-1.0
7. Save to `workspace/USER-MODEL.md` and `workspace/USER-PATTERNS.md`

**Dependencies:** audit.jsonl, session search index, USER-MODEL.md (previous model)
**Notification:** Webhook → Telegram DM

### 11. market-review
**Schedule:** Weekly Friday 10:00 UTC
**Module:** `kronos/cron/market_review.py`

Weekly investment market review:
1. Load tickers from `workspace/skills/investment-analysis/references/WATCHLIST.md`
2. Brave Search for news per ticker (freshness=past week, up to 10 tickers)
3. LLM synthesis (Sonnet standard): market overview, per-ticker events + sentiment, next week outlook, actionable recommendations
4. Send HTML report to Telegram

**Dependencies:** BRAVE_API_KEY, WATCHLIST.md
**Notification:** Bot API → Telegram Finance topic

## Scheduler Implementation

The scheduler (`kronos/cron/scheduler.py`) is a lightweight async cron without external dependencies:

- **Periodic jobs**: checked every 30 seconds, run if `interval_seconds` elapsed since last run
- **Daily jobs**: run when `hour == cron_hour` (UTC), at most once per hour
- **Weekly jobs**: additionally checks `weekday == cron_weekday` (0=Monday, 6=Sunday)
- Jobs run as `asyncio.create_task()` — non-blocking
- Initial 30-second delay after startup for bridge/webhook to be ready
- Jobs cannot overlap (flag `_running` prevents re-entry)

## Notification Methods

Two delivery methods in `kronos/cron/notify.py`:

| Method | When | How |
|--------|------|-----|
| `send_webhook()` | Simple notifications | POST to local bridge webhook (port 8788) |
| `send_bot_api()` | Topic messages | Direct Telegram Bot API (supports `message_thread_id`) |

Both support message chunking (4000 char limit) and parse_mode (HTML).
