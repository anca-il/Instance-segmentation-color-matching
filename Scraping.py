import pandas as pd
import numpy as np
import random
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options  # options for browser

from PIL import Image
import io

from bs4 import BeautifulSoup
import requests
import urllib.request

# Read the file with the links to the chosen pages
data = pd.read_csv("links.csv")

# Extract the urls and file name prefix in 2 lists
list_links = data["Images link"].to_list()
list_names = data["File_names"].to_list()

# Initialise web driver
# Add in options to not show page while parsing
options = webdriver.ChromeOptions()
options.add_argument('headless')

driver = webdriver.Chrome(options=options)


# Create functions for scraping
def get_images(url_name):
    driver.get(url_name)

    image_objects = driver.find_elements_by_class_name("grid-item--image.__loaded")

    return image_objects


def get_image_urls(images):
    image_urls = list()

    for each_image in images:
        if each_image.get_attribute('srcset') and 'http' in each_image.get_attribute('srcset'):
            image_urls.append(each_image.get_attribute('srcset'))

    image_urls = image_urls[:-1]  # delete the last image url because this is usually the designer waving at the
    # audience

    return image_urls


def change_urls_large(url_list):
    good_urls = list()

    for i in url_list:
        good_urls.append(i.replace("w_195", "w_400"))
        # changes images urls to the larger size

    return good_urls


def filter_urls(url_list):
    # randomly sample half of the images
    nr_images = int(len(url_list) / 2)

    url_sample = random.sample(url_list, nr_images)

    return url_sample


def open_images(url_name):
    time.sleep(1)

    image_content = requests.get(url_name).content
    image_file = io.BytesIO(image_content)
    image = Image.open(image_file)

    return image


def download_images(image, i, file_prefix):
    realname = file_prefix + str(i)

    file_path = "/{}.jpeg".format(realname)

    with open(file_path, 'wb') as f:
        image.save(f, "JPEG", quality=85)


# Put functions together
def process_links(url_name):
    image_objects = get_images(url_name)
    image_urls = get_image_urls(image_objects)
    good_urls = change_urls_large(image_urls)
    filtered_urls = filter_urls(good_urls)

    return filtered_urls


def process_images(image_url_list, file_prefix):
    for i in range(0, len(image_url_list)):
        url = image_url_list[i]
        image = open_images(url)
        download_images(image, i, file_prefix)


# Run scraper
N = len(list_links)

for i in range(0, N):
    time.sleep(5)
    print(i)

    current_url_name = list_links[i]
    current_file_prefix = list_names[i]

    image_urls = process_links(current_url_name)
    process_images(image_urls, current_file_prefix)

# Close webdriver
driver.quit()
