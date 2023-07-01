import requests
import time
from bs4 import BeautifulSoup
import csv

URL     = "https://pureportal.coventry.ac.uk/en/organisations/centre-for-intelligent-healthcare/publications/";
domain  = "https://pureportal.coventry.ac.uk"
data    = []

def crawl_task(current_url):
    print("Scanning .....  "+current_url)
    response = requests.get(current_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, "html.parser")

        content_elements = soup.find_all(class_="list-result-item")
        for element in content_elements:
            title_element = element.select_one("h3.title a.link span")
            title = title_element.get_text(strip=True)

            publication_link_element = element.select_one("h3.title a.link")
            publication_link = publication_link_element["href"]

            date_element = element.select_one("span.date")
            date = date_element.get_text(strip=True)

            date_element = element.select_one("span.date")
            date = date_element.get_text(strip=True)

            rcih_authors_elements = element.select("a.link.person");
            rcih_authors = [
                {"name": author.text.strip(), "link": author["href"]}
                for author in rcih_authors_elements
            ]

            # find all authors except which have links (i.e. rcih auhtors)
            rendering_portal = element.find("div", class_="rendering_portal-short")
            author_spans = rendering_portal.find_all("span", class_=False)

            authors = [
                {"name": author.text.strip(), "link": None}
                for author in author_spans
                if author.find_parent("a") is None
            ]

            data.append(
                {
                    "title": title,
                    "publication_link": publication_link,
                    "date": date,
                    "RCIH_authors": rcih_authors,
                    "authors": authors,
                }
            )

        #get next URL
        next_page = soup.select_one("nav.pages ul li.next a");
        if next_page is not None:
            crawl_task(domain + next_page['href']);
        else:
            filename = 'data.csv'
            # Extract field names from the first dictionary in the list
            fieldnames = data[0].keys()

            # Open the CSV file in write mode
            with open(filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                # Write the header row
                writer.writeheader()

                # Write the data rows
                writer.writerows(data)

            print(f"List of dictionaries saved to {filename} successfully.")
    
# crawl_task(URL)


def test_task():
    print("on the crawler.py page");
    return {"error": "sample error"};

test_task()

