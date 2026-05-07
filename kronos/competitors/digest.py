"""Competitor monitoring orchestrator — fetch, diff, synthesize digest."""

import asyncio
import logging
import os
from datetime import UTC, datetime

from langchain_core.messages import HumanMessage

from kronos.competitors.config import load_competitors
from kronos.competitors.diff import diff_snapshots
from kronos.competitors.fetchers import _REQUEST_DELAY, fetch_all_for_competitor
from kronos.competitors.models import Change, ChangeType, CompetitorConfig, Severity
from kronos.competitors.store import CompetitorStore
from kronos.llm import ModelTier, get_model

log = logging.getLogger("kronos.competitors.digest")


_PRODUCT_DESC = os.environ.get(
    "PRODUCT_DESCRIPTION",
    "your product",
)

DIGEST_PROMPT = """You are a competitive intelligence analyst for {product_desc}.

Here are changes detected across all monitoring channels in the last 24 hours:

CRITICAL:
{critical}

IMPORTANT:
{important}

INFO:
{info}

Compose a concise digest (5-10 sentences):
1. Key changes and what they mean
2. Strategic implications for our product
3. Recommended actions (if any)

Group by competitor when multiple changes come from the same one.
Format: Telegram-friendly, no markdown headers. Use emoji for readability.
Write in Russian. Keep it under 2500 chars."""


class CompetitorMonitor:
    """Orchestrates the full competitor monitoring pipeline."""

    def __init__(self) -> None:
        self.store = CompetitorStore()
        self.competitors = load_competitors()
        self.last_changes_count = 0
        self.last_competitors_checked = 0

    async def run_daily_check(self) -> str | None:
        """Run full daily check: fetch → diff → store → synthesize.

        Returns digest text, or None if this is the first run (baseline only).
        """
        if not self.competitors:
            log.warning("No competitors configured")
            return "No competitors configured."

        all_changes: list[Change] = []
        is_baseline = False
        checked = 0

        for comp in self.competitors:
            try:
                changes = await self._check_competitor(comp)
                checked += 1

                # Detect baseline (first run)
                for ch in changes:
                    if ch.change_type == ChangeType.NEW_COMPETITOR:
                        is_baseline = True

                all_changes.extend(changes)

            except Exception as e:
                log.error("Failed to check %s: %s", comp.name, e)

            # Rate limiting between competitors
            await asyncio.sleep(_REQUEST_DELAY)

        self.last_competitors_checked = checked
        self.last_changes_count = len(all_changes)

        log.info(
            "Checked %d competitors, found %d changes (baseline=%s)",
            checked, len(all_changes), is_baseline,
        )

        # First run: save baseline, no digest
        if is_baseline:
            return None

        # No meaningful changes
        real_changes = [c for c in all_changes if c.change_type != ChangeType.NEW_COMPETITOR]
        if not real_changes:
            return None

        # Synthesize digest via LLM
        digest = await self._synthesize(real_changes)

        # Save important changes to Mem0 for long-term memory
        await self._save_to_mem0(real_changes)

        return digest

    async def _check_competitor(self, comp: CompetitorConfig) -> list[Change]:
        """Check a single competitor across ALL channels (Phase 1 + Phase 2)."""
        changes: list[Change] = []

        # Upsert competitor record
        self.store.upsert_competitor(comp.id, comp.name, comp.tier, {
            "ios_id": comp.ios_id,
            "android_package": comp.android_package,
            "website": comp.website,
        })

        # --- Phase 1: App Store / Play Store ---
        snapshots = await fetch_all_for_competitor(comp.ios_id, comp.android_package)

        for channel, snapshot in snapshots.items():
            curr = snapshot.to_dict()
            prev = self.store.get_latest_snapshot(comp.id, channel)
            channel_changes = diff_snapshots(comp.id, comp.name, channel, prev, curr)
            self.store.save_snapshot(comp.id, channel, curr)
            self._persist_changes(channel_changes)
            changes.extend(channel_changes)

        # --- Phase 2: Web + Social ---
        from kronos.competitors.web_fetchers import check_all_web_channels

        try:
            web_changes = await check_all_web_channels(comp, self.store)
            self._persist_changes(web_changes)
            changes.extend(web_changes)
        except Exception as e:
            log.warning("Web channel check failed for %s: %s", comp.name, e)

        return changes

    def _persist_changes(self, changes: list[Change]) -> None:
        """Save detected changes to the database."""
        for change in changes:
            self.store.save_change(
                competitor_id=change.competitor_id,
                channel=change.channel,
                change_type=change.change_type.value,
                severity=change.severity.value,
                summary=change.summary,
                details=change.details,
            )

    async def _synthesize(self, changes: list[Change]) -> str:
        """Use LLM to synthesize human-readable digest from changes."""
        critical = [c for c in changes if c.severity == Severity.CRITICAL]
        important = [c for c in changes if c.severity == Severity.IMPORTANT]
        info = [c for c in changes if c.severity == Severity.INFO]

        prompt = DIGEST_PROMPT.format(
            product_desc=_PRODUCT_DESC,
            critical="\n".join(c.summary for c in critical) or "None",
            important="\n".join(c.summary for c in important) or "None",
            info="\n".join(c.summary for c in info) or "None",
        )

        # Add release notes for version updates
        version_updates = [c for c in changes if c.change_type == ChangeType.VERSION_UPDATE]
        if version_updates:
            notes_section = "\n\nRelease notes for version updates:"
            for c in version_updates:
                notes = c.details.get("release_notes", "")
                if notes and notes != "No notes":
                    notes_section += (
                        f"\n{c.competitor_name} "
                        f"{c.details.get('new_version', '')}: {notes[:500]}"
                    )
            prompt += notes_section

        # Add blog post details
        blog_posts = [c for c in changes if c.change_type == ChangeType.BLOG_POST]
        if blog_posts:
            blog_section = "\n\nNew blog posts:"
            for c in blog_posts:
                blog_section += f"\n{c.competitor_name}: {c.summary}"
                if c.details.get("summary"):
                    blog_section += f" — {c.details['summary'][:200]}"
            prompt += blog_section

        # Add press mention details
        press = [c for c in changes if c.change_type == ChangeType.PRESS_MENTION]
        if press:
            press_section = "\n\nPress mentions:"
            for c in press:
                press_section += f"\n{c.summary}"
            prompt += press_section

        model = get_model(ModelTier.LITE)
        response = model.invoke([HumanMessage(content=prompt)])
        digest = response.content if isinstance(response.content, str) else str(response.content)

        return digest

    async def _save_to_mem0(self, changes: list[Change]) -> None:
        """Save important/critical daily changes to Mem0 for long-term memory."""
        important = [c for c in changes if c.severity in (Severity.CRITICAL, Severity.IMPORTANT)]
        if not important:
            return

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        summary = f"Competitor changes {today}: " + "; ".join(c.summary for c in important[:10])

        try:
            from kronos.memory.store import add_memories

            messages = [
                {"role": "user", "content": f"Competitor monitoring update for {today}"},
                {"role": "assistant", "content": summary},
            ]
            add_memories(messages, user_id="competitor_monitor")
            log.info("Saved %d important changes to Mem0", len(important))
        except Exception as e:
            log.warning("Mem0 save failed (daily): %s", e)

    async def get_status_summary(self) -> str:
        """Return current status for on-demand queries."""
        competitors = load_competitors()
        tier1 = [c for c in competitors if c.tier == 1]
        tier2 = [c for c in competitors if c.tier == 2]

        # Get recent changes
        recent = self.store.get_undigested_changes()

        lines = [
            f"Monitoring {len(competitors)} competitors ({len(tier1)} tier-1, {len(tier2)} tier-2).",
            "Channels: App Store, Play Store, websites, blogs, Twitter, press, ProductHunt, jobs.",
        ]

        if recent:
            lines.append(f"\n{len(recent)} undigested changes:")
            for ch in recent[:15]:
                lines.append(f"  \u2022 [{ch['severity']}] {ch['summary']}")
            if len(recent) > 15:
                lines.append(f"  ... and {len(recent) - 15} more")
        else:
            lines.append("No recent undigested changes.")

        return "\n".join(lines)
