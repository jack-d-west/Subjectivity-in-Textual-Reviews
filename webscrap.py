import pandas as pd
import time
import random
import json

import undetected_chromedriver as uc

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup


# ---------------- DRIVER ---------------- #

def get_driver():

    options = uc.ChromeOptions()

    # Run browser off-screen instead of headless
    options.add_argument("--window-position=-32000,-32000")

    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    options.add_argument("--window-size=1920,1080")

    driver = uc.Chrome(
        options=options,
        version_main=147
    )

    # Hide webdriver flag
    driver.execute_script(
        """
        Object.defineProperty(
            navigator,
            'webdriver',
            {get: () => undefined}
        )
        """
    )

    return driver


# ---------------- LOAD COOKIES ---------------- #

def load_cookies(driver, cookie_file="cookies.json"):

    driver.get("https://www.imdb.com")

    time.sleep(3)

    with open(cookie_file, "r", encoding="utf-8") as f:
        cookies = json.load(f)

    for cookie in cookies:

        try:

            if "sameSite" in cookie:

                if cookie["sameSite"] not in ["Strict", "Lax", "None"]:
                    cookie["sameSite"] = "Lax"

            driver.add_cookie(cookie)

        except:
            pass

    driver.refresh()

    time.sleep(3)

    print("✅ Cookies loaded successfully")


# ---------------- SCRAPER ---------------- #

def scrape_imdb_reviews(movie_id, num_pages=1):

    driver = get_driver()

    try:

        load_cookies(driver)

        # Visit homepage ONCE
        driver.get("https://www.imdb.com")

        time.sleep(3)

        all_movies = []
        all_ratings = []
        all_titles = []
        all_reviews = []
        all_likes = []
        all_dislikes = []

        movie_name = None

        for rating_filter in range(1, 11):

            print(f"\n🔎 Scraping {rating_filter}-star reviews...")

            for page in range(num_pages):

                start = page * 10

                url = (
                    f"https://www.imdb.com/title/"
                    f"{movie_id}/reviews/"
                    f"?rating={rating_filter}&start={start}"
                )

                driver.get(url)

                try:

                    WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located(
                            (
                                By.CSS_SELECTOR,
                                "article.user-review-item"
                            )
                        )
                    )

                except:

                    print("⚠️ Reviews not detected. Skipping...")

                    continue

                time.sleep(random.uniform(2, 4))

                soup = BeautifulSoup(
                    driver.page_source,
                    "html.parser"
                )

                # ---------------- MOVIE NAME ---------------- #

                if movie_name is None:

                    title_block = soup.select_one(
                        '[data-testid="hero__primary-text"]'
                    )

                    movie_name = (
                        title_block.get_text(strip=True)
                        if title_block else movie_id
                    )

                    print(f"🎬 Movie detected: {movie_name}")

                # ---------------- REVIEW CARDS ---------------- #

                review_cards = soup.select(
                    "article.user-review-item"
                )

                print(f"  Found {len(review_cards)} reviews")

                for card in review_cards:

                    # TITLE
                    title_tag = card.select_one(
                        '[data-testid="review-summary"]'
                    )

                    title = (
                        title_tag.get_text(strip=True)
                        if title_tag else ""
                    )

                    # REVIEW TEXT
                    review_tag = card.select_one(
                        "div.ipc-html-content-inner-div"
                    )

                    review = (
                        review_tag.get_text(strip=True)
                        if review_tag else ""
                    )

                    # RATING
                    rating_tag = card.select_one(
                        "span.ipc-rating-star--rating"
                    )

                    rating = (
                        int(rating_tag.get_text(strip=True))
                        if rating_tag else rating_filter
                    )

                    # LIKES
                    like_tag = card.select_one(
                        ".ipc-voting__label__count--up"
                    )

                    likes = (
                        like_tag.get_text(strip=True)
                        if like_tag else "0"
                    )

                    # DISLIKES
                    dislike_tag = card.select_one(
                        ".ipc-voting__label__count--down"
                    )

                    dislikes = (
                        dislike_tag.get_text(strip=True)
                        if dislike_tag else "0"
                    )

                    # STORE DATA
                    all_movies.append(movie_name)
                    all_ratings.append(rating)
                    all_titles.append(title)
                    all_reviews.append(review)
                    all_likes.append(likes)
                    all_dislikes.append(dislikes)

                print(f"  ✅ Page {page + 1} done")

                time.sleep(random.uniform(1, 3))

        # ---------------- SAVE DATAFRAME ---------------- #

        df = pd.DataFrame({

            "movie": all_movies,

            "rating": all_ratings,

            "likes": all_likes,

            "dislikes": all_dislikes,

            "title": all_titles,

            "review": all_reviews
        })

        filename = f"{movie_id}_all_ratings_reviews.csv"

        df.to_csv(filename, index=False)

        print(f"\n💾 Saved {len(df)} reviews to {filename}")

        return df

    finally:

        driver.quit()

        print("\n🛑 Browser closed cleanly")


# ---------------- RUN ---------------- #

if __name__ == "__main__":

    df = scrape_imdb_reviews(
        "tt1375666",
        num_pages=1
    )