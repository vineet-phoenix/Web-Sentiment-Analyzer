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

async def universal_text_scraper(url: str) -> str:
    """
    Universal text-only scraper for:
    - Movie reviews
    - Product reviews
    - Social media text posts
    """

    # -----------------------------
    # 1️⃣ Selector strategy
    # -----------------------------
    if "imdb.com" in url:
        selectors = [
            "div.ipc-html-content-inner-div",
            "div[data-testid='review-card'] span"
        ]

    elif any(site in url for site in ["amazon.", "flipkart.", "myntra."]):
        selectors = [
            "span[data-hook='review-body']",
            "div.review-text-content span",
            "div.review-text"
        ]

    elif any(site in url for site in ["twitter.com", "x.com"]):
        selectors = [
            "article div[lang]"   # tweet text
        ]

    elif "reddit.com" in url:
        selectors = [
            "div[data-test-id='comment'] p",
            "div[data-click-id='text']"
        ]

    else:
        # 🔥 Generic semantic fallback
        selectors = [
            "article p",
            "section p",
            "div[role='article'] p",
            "p"
        ]

    # -----------------------------
    # 2️⃣ crawl4ai configs
    # -----------------------------
    browser_config = BrowserConfig(
        headless=True,
        browser_type="chromium"
    )

    run_config = CrawlerRunConfig(
        remove_overlay_elements=True,
        process_iframes=False,
        disable_images=True,          # 🚫 avoids image context errors
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=15,

        # 🔥 Critical
        wait_until="domcontentloaded",
        page_timeout=90_000,

        content_selectors=selectors
    )

    # -----------------------------
    # 3️⃣ Run crawl
    # -----------------------------
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)

            if not result.success:
                return f"Error scraping page: {result.error_message}"

            text = (
                result.markdown.fit_markdown
                if hasattr(result.markdown, "fit_markdown")
                else result.markdown
            )

            if not text or not text.strip():
                return "Error: Page loaded but no relevant text found."

            return text

    except Exception as e:
        return f"crawl4ai scraping failed: {e}"


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
