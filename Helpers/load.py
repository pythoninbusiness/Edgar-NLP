from bs4 import BeautifulSoup
import requests
import sqlite3
from . import textpreprocess
import pandas as pd


headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"}


def load_soup_from_file_or_edgar(report_path_or_url):
    """
    Purpose: To return the beautifulsoup object from a file path or edgar filing url
    Input: Edgar Filing url 
    Output: BS4 soup object
    """

    edgar = False
    if "http" in report_path_or_url:
        edgar = True

    if edgar:
        report_path_or_url = report_path_or_url.replace("ix?doc=/", "")
        edgar_request = requests.get(report_path_or_url, headers=headers)
        assert edgar_request.status_code == 200, f"Edgar request failed -- Code: {edgar_request.status_code}"
        soup = BeautifulSoup(edgar_request.text, 'lxml')
        
    else:    
        with open(report_path_or_url, 'r', encoding="utf8") as file_:
            html_text = file_.read()

        soup = BeautifulSoup(html_text, 'lxml')
    
    return soup


def check_database_for_duplicates(db_engine, table_name, year, period):

    cursor = db_engine.cursor()
    cursor.execute(f" SELECT * FROM {table_name} WHERE year='{year}' and period='{period}' ")
    results = cursor.fetchone()

    if results:
        return True
    else:
        return False


def append_to_database(data, database_name, table_name, year, period, link, ticker):

    """
    Purpose: Add a filing's tokenized sentences to a database for future analysis.
    Input: Dataframe and accompanying data from the report it was obtained from.

    append_to_database(
        data=df,
        database_name=database_name,
        table_name=table_name,
        year=2020,
        period="Q4",
        link="https://www.sec.gov/Archives/edgar/data/1679688/000167968821000018/clny-20201231.htm"
    )
    """

    engine = sqlite3.connect(database_name)

    if check_database_for_duplicates(engine, table_name=table_name, year=year, period=period):
        print("Entry already exists in DB.")
        return False

    data["Year"] = year
    data["Period"] = period
    data["Type"] = "10Q" if period in ["Q1", "Q2", "Q3"] else "10K"
    data["Link"] = link
    data["Ticker"] = ticker

    data.to_sql(table_name, con=engine, if_exists="append")
    engine.close()

    return True


def populate_database_from_rss_feed(database_name, table_name, rss_feed, ticker):
    """ 
    Purpose: Provide the RSS Feed; apply filters in edgar before pasting here (i.e., how many to scrape)
    EX: https://data.sec.gov/rss?cik=1679688&type=10-K,10-Q,10-KT,10-QT,NT%2010-K,NT%2010-Q,NTN%2010K,NTN%2010Q&count=5 
    """

    soup = load_soup_from_file_or_edgar(rss_feed)
    entries = soup.find_all("entry")

    mapper = {
        "03/31": "Q1", "04/00": "Q1", "05/00": "Q1", "06/00": "Q1", 
        "06/30": "Q2", "07/00": "Q2", "08/00": "Q2", "09/00": "Q2",
        "09/30": "Q3", "10/00": "Q3", "11/00": "Q3", "12/00": "Q3",
        "12/31": "Q4", "01/00": "Q4", "02/00": "Q4",  "03/00": "Q4", 
    }

    for entry in entries:
        report_link_to_filing_package = entry.find("link").get('href')
        soup_to_find_actual_report_link = BeautifulSoup(requests.get(report_link_to_filing_package, headers=headers).text, 'lxml')
        report_link_extension = soup_to_find_actual_report_link.find(class_="tableFile").find_all('tr')[1].find("a").get('href')
        report_link = "https://www.sec.gov/" + report_link_extension.replace("/ix?doc=/", "")

        if soup.find("report-date"):
            report_year = entry.find("report-date").text.split("-")[0]
            report_period = mapper["/".join(entry.find("report-date").text.split("-")[1:])]
        
        # if can't find report-date, we will use filing-date
        else:
            report_year = entry.find("filing-date").text.split("-")[0]
            report_period = mapper["/".join([entry.find("filing-date").text.split("-")[1], "00"])]
            if report_period == "Q4":
                report_year = f"{int(report_year)-1}"

        entry_df = textpreprocess.turn_filing_into_sentences_df(report_link)

        added_to_db = append_to_database(
            data=entry_df,
            database_name=database_name,
            table_name=table_name,
            year=report_year,
            period=report_period,
            link=report_link,
            ticker=ticker
        )

        if added_to_db:
            print(f"Loaded {report_period}-{report_year} to the database!")