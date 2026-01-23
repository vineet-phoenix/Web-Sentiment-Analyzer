import streamlit as st
import sys
import os
import subprocess

# --- Playwright (SYNC) ---
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_emotion import (
    predict_emotion,
    resolve_model_dir,
    load_tokenizer_and_model
)

# ------------------ MODEL LOAD ------------------

MODEL_DIR = resolve_model_dir(None)
loaded_model, loaded_tokenizer = load_tokenizer_and_model(MODEL_DIR)

if loaded_model is None or loaded_tokenizer is None:
    st.warning(
        "Model or tokenizer not found. "
        "Add model (.h5) and tokenizer.json to models/ directory."
    )

# ------------------ PLAYWRIGHT SETUP ------------------

def ensure_playwright_installed():
    try:
        subprocess.check_call(
            [sys.executable, "-m", "playwright", "install", "chromium"]
        )
    except Exception as e:
        raise RuntimeError(f"Playwright install failed: {e}")

# ------------------ SCRAPER (SYNC) ------------------

def scrape_with_playwright(url: str) -> str:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"]
            )

            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120 Safari/537.36"
                )
            )

            page = context.new_page()

            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            page.wait_for_load_state("networkidle", timeout=30000)

            # Remove overlays / modals
            page.evaluate("""
                () => {
                    document.querySelectorAll(
                        '[role="dialog"], .modal, .overlay'
                    ).forEach(e => e.remove());
                }
            """)

            # Extract readable text
            content = page.evaluate("""
                () => {
                    const tags = document.querySelectorAll(
                        'h1,h2,h3,p'
                    );
                    return Array.from(tags)
                        .map(t => t.innerText.trim())
                        .filter(Boolean)
                        .join('\\n\\n');
                }
            """)

            browser.close()

            if not content.strip():
                return "Error: Page loaded but no readable content found."

            return content

    except PlaywrightTimeout:
        return "Error: Page load timed out."
    except Exception as e:
        return f"Playwright scraping failed: {e}"

# ------------------ STREAMLIT UI ------------------

st.title("Web Content Emotion Analyzer")
st.write("Scrape a webpage using Playwright and predict its dominant emotion.")

url_input = st.text_area(
    "Enter URL",
    "https://www.imdb.com/title/tt15239678/reviews",
    height=100
)

if st.button("Analyze Emotion"):
    if not url_input:
        st.warning("Please enter a URL.")
        st.stop()

    with st.spinner("Installing Playwright (if needed)..."):
        try:
            ensure_playwright_installed()
        except Exception as e:
            st.error(e)
            st.stop()

    with st.spinner("Scraping webpage..."):
        scraped_content = scrape_with_playwright(url_input)

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
            st.error(f"Emotion prediction failed: {e}")
