# crowdfunding-analysis
Time Series Analysis on Crowdfunding Platform Kickstarter
249 projects update details section have been scraped and saved in the `scrapers/scraped_data` directory.

The `main.py` file is the main script that runs the webscraper.

The `link_scraper.py` file is the script that scrapes the links from the Kickstarter website.

The `cleanup_no_updates.py` file is the script that cleans up the scraped data by removing duplicateprojects.

The `cleanup_link_scraper.py` file is the script that cleans up the link scraper by removing duplicate links.



 scraped_data contains update section details from various projects from kickstarter's technology section projects, using this data, create and train a model analysing various components of the data ml analysis, where when backer inputs a idea or gives certain details about campaign they want to inves it, the model will give them details on whether the campaign will succeed or fail, using certain metrics obtained from model, has to be a incremental and structured approach, use  random forest to do it then can try fine tuning model using gridsearchCV or what u think is appropriate, my particualr part is coming up with metrics focussed on details on updates section of projects so no other details from projects will be scraped , but if you requuire more projects need to be scraped, i will scrape them, first explain approach u think is best after looking at scraped data in webscraper/scrapers/scraped_data, once i approve implement, not too excessive since random forest is quite simple, scraped_data contained data scraped from many projects that can be used for model