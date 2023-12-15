# WebScraping-JobsDB
 
## 1. Web scraping

<img src="https://github.com/Pisit-Janthawee/WebScraping-JobsDB/blob/main/viz_images/wordcloud.png" style="margin: 5px;" width="500" height="500">

<div style="display: flex; flex-wrap: wrap; justify-content: center;">
    <img src="https://github.com/Pisit-Janthawee/WebScraping-JobsDB/blob/main/viz_images/treemap1.png" style="margin: 5px;" width="220" height="100">
    <img src="https://github.com/Pisit-Janthawee/WebScraping-JobsDB/blob/main/viz_images/treemap2.png" style="margin: 5px;"  width="220" height="100">
    <img src="https://github.com/Pisit-Janthawee/WebScraping-JobsDB/blob/main/viz_images/treemap3.png" style="margin: 5px;"  width="220" height="100">
    <img src="https://github.com/Pisit-Janthawee/WebScraping-JobsDB/blob/main/viz_images/treemap4.png" style="margin: 5px;"  width="220" height="100">
    <img src="https://github.com/Pisit-Janthawee/WebScraping-JobsDB/blob/main/viz_images/treemap5.png" style="margin: 5px;"  width="220" height="100">
    <img src="https://github.com/Pisit-Janthawee/WebScraping-JobsDB/blob/main/viz_images/treemap6.png" style="margin: 5px;" width="220" height="100">
    <img src="https://github.com/Pisit-Janthawee/WebScraping-JobsDB/blob/main/viz_images/treemap7.png" style="margin: 5px;" width="220" height="100">
    <img src="https://github.com/Pisit-Janthawee/WebScraping-JobsDB/blob/main/viz_images/treemap8.png" style="margin: 5px;" width="220" height="100">
    <img src="https://github.com/Pisit-Janthawee/WebScraping-JobsDB/blob/main/viz_images/treemap9.png" style="margin: 5px;" width="220" height="100">
</div>

### JobsDB 
![webpage.png](https://github.com/Pisit-Janthawee/WebScraping-JobsDB/blob/main/images/website.png)](https://github.com/Pisit-Janthawee/WebScraping-JobsDB/blob/main/images/website.png)https://github.com/Pisit-Janthawee/WebScraping-JobsDB/blob/main/images/website.png)

Website: https://th.jobsdb.com/th

## 2. Objective
Need to extract data from JobsDB on a specific "Data Scientist" keyword. 

## 3. Expected outcome
spreadsheet or .csv file

## 4. Work Flow
![work_flow_img](https://github.com/Pisit-Janthawee/Web-Scraping-DrugBank-Selenium/assets/133638243/e3c8dcb8-e9ba-49ee-a58d-c0ee43e311f7)

## 5. Tool
- Python
- BeautifulSoup
- Selenium 
- Requests
- Pandas

## 6. File Description

### Folder

1. **Images**
    - *Explanation*: Image of JobsDB user-interface 
2. **viz_images**
    - *Explanation*: Image of visualization after scraping data
    - 
### 01-03 .ipynb Files

1. **01_init_notebook.ipynb**
    - *Explanation*: This initial notebook is used for scraping the data and checking missing values after scraping
    - 
2. **02_prep.ipynb**
    - *Explanation*: This prep notebook is used for preparing data after scraping, And conducting Data Extraction to make it available for visualization and analysis.
3. **03_eda.ipynb**
    - *Explanation*: This initial notebook is used for visualization and analysis. involving,
       - Top 20 Job Functions
       - Minimum Salary by Seniority (Experience level) and Position
       - A location with high employment
       - Different Roles in Data science career path with seniority (Experience Level)
       - etc.


