import app.web_scraping as scrape
import app.pinecone as pc

# data = scrape.scrape_site()
# scrape.save_data(data)
message = "Since when has States Marine Corps owned and operated Camp Lejeune as a Marine Corps base?"

pc.train_text()
pc.generate_response_streaming(message)
