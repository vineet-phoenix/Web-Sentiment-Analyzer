import streamlit as st
import asyncio
import sys
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

# fallback scraping (synchronous)
import requests
from bs4 import BeautifulSoup

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

# Import functions from predict_emotion module
from predict_emotion import predict_emotion, resolve_model_dir, load_tokenizer_and_model

# Load model and tokenizer once at startup
MODEL_DIR = resolve_model_dir(None)
loaded_model, loaded_tokenizer = load_tokenizer_and_model(MODEL_DIR)

if loaded_model is None or loaded_tokenizer is None:
    st.warning(
        "Model or tokenizer not found — predictions will be unavailable until you add the model files "
        "(model .h5 and tokenizer.json) into the models/ directory or set MODEL_DIR environment variable."
    )

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
            raise
        except FileNotFoundError as e:
            st.error(f"Playwright CLI not found (playwright package may not be installed): {e}")
            raise

def fallback_scrape(url: str) -> str:
    """
    Simple fallback scraper using requests + BeautifulSoup. Returns markdown-ish text.
    """
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        return f"Error with simple HTTP fetch for {url}: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")
    # try to extract main content: paragraphs and headings
    parts = []
    for tag in soup.find_all(["h1","h2","h3","p"]):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)
    content = "\n\n".join(parts).strip()
    if not content:
        # last resort: whole text
        content = soup.get_text(separator="\n", strip=True)
    return f"<!-- FALLBACK: simple requests scrape used -->\n\n{content}"

# Import crawl4ai components after model loading to avoid mixing responsibilities
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# Asynchronous function to scrape content
async def scrape_to_string_async(url: str) -> str:
    # Build BrowserConfig in a way that works across crawl4ai versions:
    try:
        # try to pass enable_stealth if supported
        browser_config = BrowserConfig(
            headless=True,
            enable_stealth=True,
            browser_type="chromium"
        )
    except TypeError:
        # older/newer API may not accept enable_stealth
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

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)

            if result.success:
                content = result.markdown.fit_markdown if getattr(result.markdown, "fit_markdown", None) else result.markdown
                return content
            else:
                return f"Error scraping {url}: {result.error_message}"
    except Exception as e:
        # Surface the exception and fall back to a simpler scraper.
        err_str = str(e)
        fallback_note = (
            "Playwright-style browser scraping failed with an error:\n\n"
            f"{err_str}\n\n"
            "Falling back to a simple requests+BeautifulSoup scraper (less accurate for JS-heavy sites).\n\n"
            "If you want Playwright scraping to work, ensure your host has the necessary system packages installed. "
            "On Debian/Ubuntu add the following packages to packages.txt (and redeploy) or install them on the host:\n\n"
            "libnspr4 libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libxrandr2 libxrender1 libxkbcommon0 libxss1 libasound2 libgbm1\n\n"
            "After those libs are available, Playwright Chromium should launch successfully."
        )
        # Perform fallback scraping in thread to avoid blocking event loop
        fallback_content = await asyncio.to_thread(fallback_scrape, url)
        return f"{fallback_note}\n\n---\n\n{fallback_content}"


# Streamlit app layout
st.title("Web Content Emotion Analyzer")
st.write("Enter a URL below to scrape its content and predict the dominant emotion.")

url_input = st.text_area("Enter URL:", "https://www.imdb.com/title/tt15239678/reviews", height=100)

if st.button("Analyze Emotion"):
    if url_input:
        with st.spinner("Preparing to scrape content and predict emotion..."):
            # Ensure Playwright browser binaries are installed before scraping.
            # If installation fails we'll show an error and continue to fallback scraping.
            try:
                ensure_playwright_browsers_installed()
            except Exception:
                st.warning("Playwright browser install/availability check failed; fallback scraping will be attempted.")
            # Run the scraper (async) in a background thread so Streamlit doesn't block
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(run_async_in_new_loop, scrape_to_string_async, url_input)
                scraped_content = future.result()

                st.subheader("Scraped Content (Markdown)")
                st.markdown(scraped_content)

                # Call predict_emotion with the loaded model and tokenizer
                try:
                    predicted_emotion = predict_emotion(scraped_content, loaded_model, loaded_tokenizer)
                    # predict_emotion returns an error string if model/tokenizer missing
                    if isinstance(predicted_emotion, str) and predicted_emotion.startswith("Error:"):
                        st.error(predicted_emotion)
                    else:
                        st.subheader("Predicted Emotion:")
                        st.success(predicted_emotion)
                except Exception as e:
                    st.error("Could not predict emotion due to scraping or prediction error.")
                    st.write(f"Prediction error: {e}")
    else:
        st.warning("Please enter a URL.")
