from kronos.competitors.digest import DIGEST_PROMPT


def test_competitor_digest_prompt_formats_change_sections():
    prompt = DIGEST_PROMPT.format(
        product_desc="KAOS",
        critical="Critical change",
        important="Important change",
        info="Info change",
    )

    assert "competitive intelligence analyst for KAOS" in prompt
    assert "Critical change" in prompt
    assert "Important change" in prompt
    assert "Info change" in prompt
