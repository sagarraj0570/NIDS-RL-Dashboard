from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

APP_URL = "https://nids-capstone-vit.streamlit.app/"

options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=options)

try:
    print("Opening Streamlit App...")
    driver.get(APP_URL)

    # Wait 30 seconds so Streamlit can fully load
    time.sleep(30)

    print("Visit completed successfully!")

finally:
    driver.quit()
