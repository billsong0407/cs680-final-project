import argparse
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler

def download_images_from_google(query, num_images, image_size=None):
    google_crawler = GoogleImageCrawler(storage={'root_dir': 'images'})
    google_crawler.crawl(keyword=query, max_num=num_images, 
                       filters={'size': image_size} if image_size else None)

def download_images_from_bing(query, num_images, image_size=None):
    bing_crawler = BingImageCrawler(storage={'root_dir': 'dataset/bing_images'})
    bing_crawler.crawl(keyword=query, max_num=num_images, 
                       filters={'size': image_size} if image_size else None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images from Bing based on a search query.")

    parser.add_argument('-q', '--query', type=str, required=True, help='Search query (e.g., "plant diseases")')
    parser.add_argument('-b', '--browser', type=str, required=True, help='Browser options')
    parser.add_argument('-n', '--num_images', type=int, required=True, help='Number of images to download')
    parser.add_argument('-s', '--size', type=str, choices=['small', 'medium', 'large'], 
                        help='Size of the images (optional: "small", "medium", "large")')

    args = parser.parse_args()

    if args.browser.lower() == 'chrome':
        download_images_from_google(args.query, args.num_images, args.size)
    elif args.browser.lower() == 'bing':
        download_images_from_bing(args.query, args.num_images, args.size)
    else:
        print("Unrecongnized Browser")