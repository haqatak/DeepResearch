from playwright.sync_api import sync_playwright, Playwright

def run(playwright: Playwright):
    chromium = playwright.chromium
    try:
        print("Launching browser...")
        browser = chromium.launch(headless=True)
        print("Browser launched successfully.")
    except Exception as e:
        print(f"Failed to launch browser: {e}")
        print("="*80)
        print("!! Playwright browser not found !!")
        print("Please run the following command to install the necessary browsers:")
        print("\n    playwright install\n")
        print("="*80)
        return

    page = browser.new_page()
    try:
        print("Navigating to http://iamai.no ...")
        page.goto("http://iamai.no")
        title = page.title()
        print(f"Successfully fetched page. Title: '{title}'")
    except Exception as e:
        print(f"Failed to navigate to page: {e}")
    finally:
        print("Closing browser.")
        browser.close()

with sync_playwright() as playwright:
    run(playwright)
