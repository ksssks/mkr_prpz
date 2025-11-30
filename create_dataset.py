from serpapi import GoogleSearch
import requests
import os
import time
import imghdr

API_KEY = "44e9b8fe9bd4d987d15851ab9698e0725e87d071dda582ce506641cbc5df8e6d"

BRANDS = {
    "mcdonalds": [

        "mcdonald's food packaging",
        "mcdonald's burger box logo",
        "mcdonald's fries packaging logo",
        "mcdonald's drink cup logo",
        "mcdonald's soda cup logo",
        "mcdonald's takeaway bag logo",
        "mcdonald's paper bag logo",
        "mcdonald's meal packaging",
        "mcdonald's big mac box logo",
        "mcdonald's nuggets box logo",
        "mcdonald's coffee cup logo",
        "mcdonald's mccafe cup logo",
        "mcdonald's delivery bag",
        "mcdonald's brown bag with logo",
        "mcdonald's ketchup packet logo",
        "mcdonald's packaging on table",
        "mcdonald's food on tray logo",


        "people holding mcdonald's bag",
        "people holding mcdonald's drink",
        "people holding mcdonald's fries",
        "person eating mcdonald's burger",
        "people with mcdonald's food",
        "mcdonald's food in hand",


        "mcdonald's tray food",
        "mcdonald's burger restaurant table",
        "mcdonald's menu board logo",
        "mcdonald's counter food",
        "mcdonald's self order kiosk screen menu",


        "mcdonald's food advertisement",
        "mcdonald's poster burger fries",
        "mcdonald's billboard food ad",
        "mcdonald's marketing burger poster",
        "mcdonald's outdoor food banner",


        "mcdonald's delivery rider bag",
        "mcdonald's delivery packaging",
        "mcdonald's uber eats bag",
        "mcdonald's courier backpack",

        "close up mcdonald's fries packaging",
        "close up mcdonald's burger box",
        "close up mcdonald's coffee cup logo",
        "close up mcdonald's drink cup",


        "mcdonald's top view meal",
        "mcdonald's food flat lay",
        "mcdonald's meal tray top down",
        "mcdonald's packaging top view"
    ]
}

SAVE_DIR = "logos"
IMAGES_PER_BRAND = 200
IMAGES_PER_QUERY = 5

os.makedirs(SAVE_DIR, exist_ok=True)

VALID_FORMATS = {"jpeg"}

def get_extension_from_content_type(ct):
    if not ct:
        return None
    if "jpeg" in ct:
        return "jpg"
    return None

for brand, queries in BRANDS.items():

    print(f"\n=== Collecting images for: {brand} ===")

    brand_dir = os.path.join(SAVE_DIR, brand)
    os.makedirs(brand_dir, exist_ok=True)

    downloaded = 0
    query_count = len(queries)

    while downloaded < IMAGES_PER_BRAND:

        for query in queries:

            if downloaded >= IMAGES_PER_BRAND:
                break

            print(f"\n➡️ Query: {query}")

            page = 0
            taken_from_this_query = 0

            while taken_from_this_query < IMAGES_PER_QUERY and downloaded < IMAGES_PER_BRAND:

                params = {
                    "engine": "google",
                    "q": query,
                    "tbm": "isch",
                    "ijn": page,
                    "api_key": API_KEY
                }

                search = GoogleSearch(params)
                results = search.get_dict()
                images = results.get("images_results", [])

                if not images:
                    page += 1
                    time.sleep(1)
                    continue

                for img in images:

                    if taken_from_this_query >= IMAGES_PER_QUERY:
                        break
                    if downloaded >= IMAGES_PER_BRAND:
                        break

                    url = img.get("original")
                    if not url:
                        continue

                    try:
                        r = requests.get(url, timeout=10)
                        if r.status_code != 200:
                            continue

                        ext = get_extension_from_content_type(r.headers.get("Content-Type"))
                        if ext != "jpg":
                            continue

                        filename = f"{downloaded + 1}.jpg"
                        path = os.path.join(brand_dir, filename)

                        with open(path, "wb") as f:
                            f.write(r.content)

                        if imghdr.what(path) not in VALID_FORMATS:
                            os.remove(path)
                            continue

                        print("Saved:", filename)
                        downloaded += 1
                        taken_from_this_query += 1

                    except:
                        pass

                page += 1
                time.sleep(1)

    print(f"\nDone: collected {downloaded} JPG images for {brand}.")
