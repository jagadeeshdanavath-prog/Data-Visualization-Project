
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://books.toscrape.com/catalogue/"
START_URL = "https://books.toscrape.com/catalogue/page-1.html"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def scrape_page(url: str) -> list[dict]:
    """Scrape all books from a single catalogue page."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    books = []
    for article in soup.select("article.product_pod"):
        title = article.h3.a["title"]
        price = article.select_one("p.price_color").text.strip().replace("Â", "")
        rating_word = article.p["class"][1]          # e.g. "Three"
        rating = RATING_MAP.get(rating_word, 0)
        availability = article.select_one("p.availability").text.strip()

        books.append({
            "Title":        title,
            "Price (£)":   float(price.replace("£", "")),
            "Rating":       rating,
            "Availability": availability,
        })
    return books


def get_next_page(url: str) -> str | None:
    """Return the URL of the next page, or None if we're on the last page."""
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    next_btn = soup.select_one("li.next > a")
    if next_btn:
        return BASE_URL + next_btn["href"]
    return None


def main():
    all_books = []
    url = START_URL
    page = 1
    max_pages = 5          # scrape first 5 pages (~100 books) to keep it quick

    print("=== CodeAlpha Task 1: Web Scraping ===\n")
    while url and page <= max_pages:
        print(f"  Scraping page {page}: {url}")
        books = scrape_page(url)
        all_books.extend(books)
        url = get_next_page(url)
        page += 1
        time.sleep(0.5)    # polite crawl delay

    df = pd.DataFrame(all_books)
    output_file = "books_dataset.csv"
    df.to_csv(output_file, index=False)

    print(f"\n✅ Scraped {len(df)} books from {page - 1} pages.")
    print(f"📁 Dataset saved to: {output_file}\n")
    print("--- Sample Data (first 5 rows) ---")
    print(df.head().to_string(index=False))

    print("\n--- Quick Stats ---")
    print(f"  Average Price : £{df['Price (£)'].mean():.2f}")
    print(f"  Average Rating: {df['Rating'].mean():.2f} / 5")
    print(f"  In Stock      : {(df['Availability'] == 'In stock').sum()} books")


if __name__ == "__main__":
    main()