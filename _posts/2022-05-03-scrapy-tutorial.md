---
layout: post
title:  "Scraping IMDB using Scrapy"
categories: blog assignment
permalink: posts/scrapy-tutorial
author: Pei Xi Kwok
---

Let's learn how to use webscraping to answer the following question:
>What movie or TV shows share actors with your favorite movie or show?

## 1. Setup
### 1.1 Locate the starting IMDB page
Pick your favorite movie or TV show, and locate its IMDB page. For example, my favorite movie is Blade Runner 2049. Its IMDB page is at:

```python
https://www.imdb.com/title/tt1856101/
```

### 1.2 Initialize your project
1. Create a new GitHub repository, and sync it with GitHub Desktop. This repository will house your scraper. You should commit and push each time you make significant changes to your code.
2. Open a terminal in the location of your repository on your laptop, and type:
```python
conda activate PIC16B
scrapy startproject IMDB_scraper
cd IMDB_scraper
```

### 1.3 Tweak settings
For now, add the following line to the file settings.py:
```python
CLOSESPIDER_PAGECOUNT = 20
```
This line just prevents your scraper from downloading too much data while you’re still testing things out. You’ll remove this line later.

## 2. Writing your scraper
We start by creating a file inside the spiders directory called imdb_spider.py and importing the necessary packages.
```python
import scrapy
```
To scrape a website, we need to write a spider that will crawl a site and extract the data. Spiders are classes that you will have to define to make the initial request to the site, follow links in the pages and parse the downloaded page content to extract data.

In our case, we are interested in scraping data from an actor's page based on their involvement in a specific movie. We first need to define methods that will enable us to follow links from the initial movie's IMDB page to get to the Cast & Crew page:

```python
 def parse(self,response):
        """
        starts on a movie page and navigates to Cast & Crew page

        response - response object from scrapy

        yields a scrapy request to parse Cast & Crew page
        """
        # create url for Cast & Crew page
        full_credit_url = response.url+"fullcredits"

        # navigates to Cast & Crew page
        # calls second method: parse_full_credits
        yield scrapy.Request(full_credit_url,callback=self.parse_full_credits)
```

You will notice that we have called a second function, parse_full_credits in the callback method of our scrapy.Request. It will help us follow the links on the Cast & Crew page to get to an individual actor's page. Let's define it:

```python
def parse_full_credits(self,response):
        """
        starts on the Cast & Crew page of a movie and navigates to each actor's page

        response - response object from scrapy

        yields a scrapy request to parse each actor's page
        """
        # create list of suburls of actor pages
        actor_url_list = [a.attrib["href"] for a in response.css("td.primary_photo a")]

        # for each actor with a suburl
        for url in actor_url_list:
            # create full url
            actor_url = response.urljoin(url)
            # navigate to actor page
            # call parse_actor_page method
            yield scrapy.Request(actor_url,callback=self.parse_actor_page)
```

Again, you will notice that a third function parse_actor_page is being called in the callback method of our scrapy.Request. Now that we have figured out how to follow links to navigate from the movie page to the actor page, this will help us extract the data that we want from an actor's page. In this case, we are interested in every movie or show that an actor has been involved in. 

In the method below, we will use CSS selectors to help us extract the actor's name, and the name of the movie or show they were involved in. 

```python
def parse_actor_page(self,response):
        """
        Goes through an actor's credits and returns all shows/movies they appeared in

        response - response object from scrapy

        yields dictionary containing all shows/movies they appeared in
        """
        # get actor name
        actor_name = response.css('div#name-overview-widget h1.header span::text').get()
        
        # get show/movie name
        for row in response.css('div#filmography div.filmo-row'):
            movie_or_TV_name = row.css("b a::text").get()
            yield{"actor" : actor_name, "movie_or_TV_name" : movie_or_TV_name}
```

Now that you've written all three of your parse methods, you need to combine them under one spider class in your imdb_spider.py file. This will enable the methods to be used by the spider to crawl and extract data from the IMDB website.

```python
class ImdbSpider(scrapy.Spider):
  # spider classes need to subclass scrapy.Spider

    name = 'imdb_spider'
    # identifies the Spider
    
    # starts at movie url
    start_urls = ['https://www.imdb.com/title/tt1856101/']

    def parse(self,response):
        """
        starts on a movie page and navigates to Cast & Crew page

        response - response object from scrapy

        yields a scrapy request to parse Cast & Crew page
        """
        # create url for Cast & Crew page
        full_credit_url = response.url+"fullcredits"

        # navigates to Cast & Crew page
        # calls parse_full_credits method
        yield scrapy.Request(full_credit_url,callback=self.parse_full_credits)

    def parse_full_credits(self,response):
        """
        starts on the Cast & Crew page of a movie and navigates to each actor's page

        response - response object from scrapy

        yields a scrapy request to parse each actor's page
        """
        # create list of suburls of actor pages
        actor_url_list = [a.attrib["href"] for a in response.css("td.primary_photo a")]

        # for each actor with a suburl
        for url in actor_url_list:
            # create full url
            actor_url = response.urljoin(url)
            # navigate to actor page
            # call parse_actor_page method
            yield scrapy.Request(actor_url,callback=self.parse_actor_page)

    def parse_actor_page(self,response):
        """
        Goes through an actor's credits and returns all shows/movies they appeared in

        response - response object from scrapy

        yields dictionary containing all shows/movies they appeared in
        """
        # get actor name
        actor_name = response.css('div#name-overview-widget h1.header span::text').get()
        
        # get show/movie name
        for row in response.css('div#filmography div.filmo-row'):
            movie_or_TV_name = row.css("b a::text").get()
            yield{"actor" : actor_name, "movie_or_TV_name" : movie_or_TV_name}
```
## 3. Test your scraper
Now that you've written your scraper, you can comment out the line
```python
CLOSESPIDER_PAGECOUNT = 20
```
in the settings.py file. Then, run this
```python
scrapy crawl imdb_spider -o results.csv
```
in a terminal. This will run the spider and save  a CSV file called results.csv, with columns for actor names and the movies and TV shows on which they worked. Make sure to navigate to the directory that your spiders folder is in before running this!

Here are the top 100 results of running the scraper on the Blader Runner IMDB page.
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Show or movie name</th>
      <th>Shared actors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Blade Runner 2049</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Entertainment Tonight</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Celebrity Page</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Made in Hollywood</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Extra with Billy Bush</td>
      <td>10</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Jack and Marilyn</td>
      <td>3</td>
    </tr>
    <tr>
      <th>96</th>
      <td>The Rosie O'Donnell Show</td>
      <td>3</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Izzy Gets the Fuck Across Town</td>
      <td>3</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Close Up</td>
      <td>3</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Cannes Film Festival</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 2 columns</p>
</div>

Here is a visualization of the top 5 productions that share the most actors with Blade Runner.
![hw2.png](/images/hw2.png)

You can check out the GitHub repository for this project here: https://github.com/kwpexi/movies

