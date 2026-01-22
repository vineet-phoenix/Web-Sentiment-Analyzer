import streamlit as st
import asyncio
import sys
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_async_in_new_loop(coro, *args, **kwargs):
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro(*args, **kwargs))
    finally:
        loop.close()

# Add the current directory to the system path to import predict_emotion.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_emotion import predict_emotion # Assuming predict_emotion is in predict_emotion.py
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

def ensure_playwright_browsers_installed():
    """
    Install Playwright browsers (Chromium) if they are missing.
    This runs `python -m playwright install chromium` and will only run
    when the ms-playwright cache directory is missing or empty.
    """
    cache_dir = os.path.expanduser("~/.cache/ms-playwright")
    try:
        cache_exists = os.path.exists(cache_dir) and any(os.scandir(cache_dir))
    except Exception:
        cache_exists = False

    if not cache_exists:
        # Provide UI feedback. Note: this may take some time to download the browser.
        st.info("Playwright browser binaries not found. Installing Chromium (this may take a minute)...")
        try:
            subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
            st.success("Playwright Chromium installed successfully.")
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to install Playwright browsers: {e}")
            # Re-raise so caller can decide how to proceed (we don't silently continue)
            raise
        except FileNotFoundError as e:
            st.error(f"Playwright CLI not found (playwright package may not be installed): {e}")
            raise

# Asynchronous function to scrape content
async def scrape_to_string_async(url: str) -> str:
    browser_config = BrowserConfig(
        headless=False,
        enable_stealth=True,
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

        if result.success:
            content = result.markdown.fit_markdown if result.markdown.fit_markdown else result.markdown
            return content
        else:
            return f"Error scraping {url}: {result.error_message}"


# Streamlit app layout
st.title("Web Content Emotion Analyzer")
st.write("Enter a URL below to scrape its content and predict the dominant emotion.")

# Changed st.text_input to st.text_area
url_input = st.text_area("Enter URL:", "https://www.imdb.com/title/tt15239678/reviews", height=100)

if st.button("Analyze Emotion"):
    if url_input:
        with st.spinner("Preparing to scrape content and predict emotion..."):
            # Ensure Playwright browser binaries are installed before scraping
            try:
                ensure_playwright_browsers_installed()
            except Exception:
                st.error("Playwright browser installation failed. Aborting scraping.")
                # stop here to avoid further errors
            else:
                with st.spinner("Scraping content and predicting emotion..."):
                    # use in Streamlit where you previously called asyncio.run(...)
                    with ThreadPoolExecutor(max_workers=1) as ex:
                        future = ex.submit(run_async_in_new_loop, scrape_to_string_async, url_input)
                        scraped_content = future.result()
                        st.subheader("Scraped Content (Markdown)")
                        st.markdown(scraped_content)

                        if "Error scraping" not in scraped_content:
                            predicted_emotion = predict_emotion(scraped_content)
                            st.subheader("Predicted Emotion:")
                            st.success(predicted_emotion)
                        else:
                            st.error("Could not predict emotion due to scraping error.")
    else:
        st.warning("Please enter a URL.")
