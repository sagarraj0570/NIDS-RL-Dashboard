from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

# Your Streamlit URL
URL = "https://nids-capstone-vit.streamlit.app/"

options = Options()

options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")

driver = webdriver.Chrome(options=options)

driver.get(URL)

# Wait for Streamlit to fully establish websocket
time.sleep(20)

driver.quit()

print("Visited successfully.")
