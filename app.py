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

async def crawl4ai_scrape(url: str) -> str:
    try:
        browser_config = BrowserConfig(
            headless=True,
            browser_type="chromium"
        )

        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=20,
            remove_overlay_elements=True,
            process_iframes=True,

            # 🔥 IMPORTANT FIXES
            wait_until="domcontentloaded",   # ❌ NOT networkidle
            page_timeout=90_000
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)

            if not result.success:
                return f"Error scraping page: {result.error_message}"

            content = (
                result.markdown.fit_markdown
                if hasattr(result.markdown, "fit_markdown")
                else result.markdown
            )

            if not content or not content.strip():
                return "Error: Page loaded but no readable content extracted."

            return content

    except Exception as e:
        return f"crawl4ai failed: {e}"

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
            crawl4ai_scrape(url)
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
