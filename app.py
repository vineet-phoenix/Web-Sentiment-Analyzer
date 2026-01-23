import streamlit as st
import asyncio
import sys
import os
import subprocess

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode
)

# -------------------------------------------------
# 🔁 GLOBAL EVENT LOOP (CRITICAL FIX)
# -------------------------------------------------

@st.cache_resource
def get_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop

loop = get_event_loop()

# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_emotion import (
    predict_emotion,
    resolve_model_dir,
    load_tokenizer_and_model
)

MODEL_DIR = resolve_model_dir(None)
loaded_model, loaded_tokenizer = load_tokenizer_and_model(MODEL_DIR)

if loaded_model is None or loaded_tokenizer is None:
    st.warning("Model or tokenizer not found.")

# -------------------------------------------------
# PLAYWRIGHT INSTALL (SAFE)
# -------------------------------------------------

def ensure_playwright():
    subprocess.check_call(
        [sys.executable, "-m", "playwright", "install", "chromium"]
    )

# -------------------------------------------------
# CRAWL4AI SCRAPER (ASYNC, STABLE)
# -------------------------------------------------
async def crawl4ai_fetch(url: str) -> str:
    browser_config = BrowserConfig(
        headless=True,
        browser_type="chromium"
    )

    run_config = CrawlerRunConfig(
        remove_overlay_elements=True,
        process_iframes=False,
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=10,

        # IMPORTANT: safe args only
        wait_until="domcontentloaded",
        page_timeout=90_000
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)

        if not result.success:
            return f"Error scraping page: {result.error_message}"

        text = (
            result.markdown.fit_markdown
            if hasattr(result.markdown, "fit_markdown")
            else result.markdown
        )

        return text
        
def extract_relevant_text(raw_text: str, url: str) -> str:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    filtered = []

    if "imdb.com" in url:
        for l in lines:
            if len(l) > 40 and not l.lower().startswith("helpful"):
                filtered.append(l)

    elif any(site in url for site in ["amazon.", "flipkart.", "myntra."]):
        for l in lines:
            if len(l) > 30 and "out of 5" not in l.lower():
                filtered.append(l)

    elif any(site in url for site in ["twitter.com", "x.com"]):
        for l in lines:
            if len(l) > 15 and not l.startswith("@"):
                filtered.append(l)

    elif "reddit.com" in url:
        for l in lines:
            if len(l) > 20 and not l.lower().startswith("level"):
                filtered.append(l)

    else:
        for l in lines:
            if len(l) > 40:
                filtered.append(l)

    # Deduplicate
    seen = set()
    unique = []
    for l in filtered:
        if l not in seen:
            unique.append(l)
            seen.add(l)

    return "\n\n".join(unique)

async def universal_review_scraper(url: str) -> str:
    raw = await crawl4ai_fetch(url)

    if raw.startswith("Error"):
        return raw

    clean = extract_relevant_text(raw, url)

    if not clean.strip():
        return "Error: No review or post text found."

    return clean

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------

st.title("Web Content Emotion Analyzer (crawl4ai)")
st.write("JS-aware scraping using crawl4ai + Playwright")

url = st.text_area(
    "Enter URL",
    "https://www.imdb.com/title/tt15239678/reviews",
    height=100
)

if st.button("Analyze Emotion"):
    if not url:
        st.warning("Please enter a URL.")
        st.stop()

    with st.spinner("Ensuring Playwright is available..."):
        try:
            ensure_playwright()
        except Exception as e:
            st.error(f"Playwright install failed: {e}")
            st.stop()

    with st.spinner("Scraping webpage with crawl4ai..."):
        scraped_content = loop.run_until_complete(
    universal_review_scraper(url)
)
    if scraped_content.startswith("Error"):
        st.error(scraped_content)
        st.stop()

    st.subheader("Scraped Content")
    st.markdown(scraped_content)

    with st.spinner("Predicting emotion..."):
        try:
            emotion = predict_emotion(
                scraped_content,
                loaded_model,
                loaded_tokenizer
            )

            if isinstance(emotion, str) and emotion.startswith("Error"):
                st.error(emotion)
            else:
                st.subheader("Predicted Emotion")
                st.success(emotion)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
