import streamlit as st
import asyncio
import sys
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_async_in_new_loop(coro, *args, **kwargs):
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro(*args, **kwargs))
    finally:
        loop.close()

# Add the current directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_emotion import predict_emotion, resolve_model_dir, load_tokenizer_and_model

# Load model and tokenizer once
MODEL_DIR = resolve_model_dir(None)
loaded_model, loaded_tokenizer = load_tokenizer_and_model(MODEL_DIR)

if loaded_model is None or loaded_tokenizer is None:
    st.warning(
        "Model or tokenizer not found — predictions will be unavailable until you add the model files "
        "(model .h5 and tokenizer.json) into the models/ directory or set MODEL_DIR environment variable."
    )

def ensure_playwright_browsers_installed():
    cache_dir = os.path.expanduser("~/.cache/ms-playwright")
    try:
        cache_exists = os.path.exists(cache_dir) and any(os.scandir(cache_dir))
    except Exception:
        cache_exists = False

    if not cache_exists:
        st.info("Installing Playwright Chromium (this may take a minute)...")
        subprocess.check_call(
            [sys.executable, "-m", "playwright", "install", "chromium"]
        )
        st.success("Playwright Chromium installed.")

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def scrape_to_string_async(url: str) -> str:
    try:
        try:
            browser_config = BrowserConfig(
                headless=True,
                enable_stealth=True,
                browser_type="chromium"
            )
        except TypeError:
            browser_config = BrowserConfig(
                headless=True,
                browser_type="chromium"
            )

        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=10,
            remove_overlay_elements=True,
            process_iframes=True
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)

            if not result.success:
                return f"Error scraping {url}: {result.error_message}"

            content = (
                result.markdown.fit_markdown
                if getattr(result.markdown, "fit_markdown", None)
                else result.markdown
            )

            if not content or not content.strip():
                return "Error: Scraping succeeded but no content was extracted."

            return content

    except Exception as e:
        return (
            "Playwright scraping failed.\n\n"
            f"Error details:\n{e}\n\n"
            "Ensure all required system dependencies for Chromium are installed."
        )

# Streamlit UI
st.title("Web Content Emotion Analyzer")
st.write("Enter a URL to scrape its content and predict the dominant emotion.")

url_input = st.text_area(
    "Enter URL:",
    "https://www.imdb.com/title/tt15239678/reviews",
    height=100
)

if st.button("Analyze Emotion"):
    if not url_input:
        st.warning("Please enter a URL.")
    else:
        with st.spinner("Scraping content and predicting emotion..."):
            try:
                ensure_playwright_browsers_installed()
            except Exception as e:
                st.error(f"Playwright setup failed: {e}")
                st.stop()

            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    run_async_in_new_loop,
                    scrape_to_string_async,
                    url_input
                )
                scraped_content = future.result()

            if scraped_content.startswith("Error"):
                st.error(scraped_content)
                st.stop()

            st.subheader("Scraped Content (Markdown)")
            st.markdown(scraped_content)

            try:
                predicted_emotion = predict_emotion(
                    scraped_content,
                    loaded_model,
                    loaded_tokenizer
                )

                if isinstance(predicted_emotion, str) and predicted_emotion.startswith("Error"):
                    st.error(predicted_emotion)
                else:
                    st.subheader("Predicted Emotion")
                    st.success(predicted_emotion)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

