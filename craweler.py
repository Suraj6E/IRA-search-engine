from apscheduler.schedulers.background import BackgroundScheduler
import requests
from bs4 import BeautifulSoup

def crawl_task():
    # Specify the URL you want to crawl
    url = 'https://example.com'

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find and extract the data you need from the parsed HTML
        # Example: Get all the links on the page
        links = []
        for link in soup.find_all('a'):
            links.append(link.get('href'))

        # Process or store the extracted data as desired
        print(links)

def schedule_crawl_task():
    # Create a scheduler instance
    scheduler = BackgroundScheduler()

    # Schedule the crawl_task to run every week (e.g., every Monday at 8:00 AM)
    scheduler.add_job(crawl_task, 'cron', day_of_week='mon', hour=8, minute=0)

    # Start the scheduler
    scheduler.start()

# Register the before_first_request function to run before the first request
@app.before_first_request
def setup_scheduler():
    schedule_crawl_task()
