import streamlit as st
import asyncio
import sys
import os
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
