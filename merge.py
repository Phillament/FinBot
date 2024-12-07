import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from html_table_parser.parser import HTMLTableParser
from bs4 import BeautifulSoup
import pandas as pd
import time
from groq import Groq
import pdfkit
import os
import openai
# from dotenv import load_dotenv

# load_dotenv()

# Initialize Groq client with your API key
GROQ_API_KEY = st.secrets["GROQ_API_KEY"] # Replace with your actual API key
client = Groq(api_key=GROQ_API_KEY)

openai.api_key = st.secrets["GPT_API_KEY"]
model_name = "gpt-4o-mini"
temperature = 0.0

# Set up the Streamlit UI
st.title("Financial Analysis Application")
st.write("Enter the organization's name and URL to start the analysis.")

# Input fields for organization name and URL
org_name = st.text_input("Organization Name")
url = st.text_input("URL")

def read_csv_as_string(file_path):
    try:
        df = pd.read_csv(file_path)
        return df.to_csv(index=False)
    except FileNotFoundError:
        return "File not found."

# Define function to scrape news data
def scrape_news_data(org: str, url: str):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-extensions")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    options.add_argument("--disable-features=SameSiteByDefaultCookies,CookiesWithoutSameSiteMustBeSecure")

    # driver = webdriver.Chrome(options=options)
    # driver = webdriver.Chrome(executable_path='E:/FinancialBot/chromedriver-win64/chromedriver.exe', options=options)
    service = Service(executable_path='E:/FinancialBot/chromedriver-win64/chromedriver.exe')
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)

    suffix_url = url.split('/')[-4:]
    scrape_url = f'https://trendlyne.com/latest-news/only-news/{suffix_url[0]}/{suffix_url[1]}/{suffix_url[2]}/'
    driver.get(scrape_url)
    time.sleep(2)  # Letting the page load

    html_content = driver.page_source
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extracting the specific news section
    news_div = soup.find('div', class_='col-md-7 col-xs-12 stock-news')
    
    # Initialize list for storing news items
    news_list = []

    if news_div:
        # Extract all post-head-subtext items with dates
        card_headers = news_div.find_all('div', class_='post-head-subtext')
        date_dict = {}
        
        for card_header in card_headers:
            # Extract the date
            date_tag = card_header.find('span')
            date = date_tag.get_text(strip=True) if date_tag else "No date"
            
            # Extract additional info from post-head-subtext
            header_info = card_header.get_text(strip=True).replace(date, "").strip()
            date_dict[date] = header_info
        
        # Extract all card-block items
        news_items = news_div.find_all('div', class_='card-block')

        # Iterate through all news items and match with date
        for item in news_items:
            # Initialize news_data here
            news_data = {}
            
            # Find the associated date for this news item
            date_tag = item.find_previous('div', class_='post-head-subtext').find('span') if item.find_previous('div', 'post-head-subtext') else None
            news_data['date'] = date_tag.get_text(strip=True) if date_tag else "No date"
            
            # Get the headline and link
            headline_tag = item.find('a', class_='newslink')
            if headline_tag:
                news_data['headline'] = headline_tag.get_text(strip=True)
                news_data['link'] = headline_tag['href']

            # Get the summary (if available)
            summary_tag = item.find('article')
            news_data['summary'] = summary_tag.get_text(strip=True) if summary_tag else "No summary"

            # Get the publication source
            source_tag = item.find('div', class_='rsssource')
            news_data['source'] = source_tag.get_text(strip=True) if source_tag else "No source"
            
            # Add post-head-subtext additional info if available
            news_data['header_info'] = date_dict.get(news_data['date'], "No header info")
            
            # Add news_data to the list if headline exists
            if 'headline' in news_data:
                news_list.append(news_data)
        
        # Create a DataFrame from the news list
        if news_list:
            df = pd.DataFrame(news_list)
            
            # Save as CSV
            csv_filename = f'{org}_news.csv'
            df.to_csv(f'{csv_filename}', index=False)
            st.write(f"Saved news data to {csv_filename}")
            st.dataframe(df)  # Display the DataFrame in Streamlit
        
        else:
            st.write("No news items found.")
    
    else:
        st.write("No news div found on the page.")

    driver.quit()
    
def scrape_quantitative_data(org: str, url: str):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-extensions")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    options.add_argument("--disable-features=SameSiteByDefaultCookies,CookiesWithoutSameSiteMustBeSecure")

    # driver = webdriver.Chrome(options=options)
    # driver = webdriver.Chrome(executable_path='E:/FinancialBot/chromedriver-win64/chromedriver.exe', options=options)
    service = Service(executable_path='E:/FinancialBot/chromedriver-win64/chromedriver.exe')
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)


    tables_type = ['quarterly-results', 'annual-results', 'balance-sheet', 'financial-ratios', 'cash-flow']
    suffix_url = url.split('/')[-4:]

    for table_type in tables_type:
        scrape_url = f'https://trendlyne.com/fundamentals/{table_type}/{suffix_url[0]}/{suffix_url[1]}/{suffix_url[2]}/'
        driver.get(scrape_url)
        time.sleep(2)  # Letting the page load

        html_content = driver.page_source
        parser = HTMLTableParser()
        parser.feed(html_content)

        if parser.tables:
            df = pd.DataFrame(parser.tables[0])
            df.columns = df.iloc[0]
            df = df[1:]
            df.to_csv(f'{org}_{table_type}.csv', index=False)
            st.write(f"Saved data to {org}_{table_type}.csv")
            st.dataframe(df) 
        else:
            st.write(f"No tables found on page {scrape_url}")

    driver.quit()

import json
import seaborn as sns
import matplotlib.pyplot as plt

def create_html_report(json_data: dict, overview:str, quant_summary: str, quant_sentiment: str, news_summary: str, news_sentiment: str):
    
    def clean_quant_summary(quant_summary: str) -> str:
        # Replace unwanted characters such as â‚¹
        cleaned_summary = quant_summary.replace('â‚¹', '₹')  # Replace incorrect rupee symbol
        cleaned_summary = cleaned_summary.replace('â€”', '—')  # Replace any incorrect dash symbols
        return cleaned_summary.strip()  # Remove any leading/trailing whitespace
    
    quant_summary = clean_quant_summary(quant_summary)
    
    company_name = json_data["company_name"]
    about_company = json_data["about_company"]
    key_financials = json_data["key_financial_data_and_values"]
    financial_ratios = json_data["financial_ratios_analysis"]
    investment_sentiment = json_data["investment_sentiment"]
    five_year_summary = json_data["five_year_summary_fundamentals"]
    # comparable_companies = json_data["comparable_companies_and_trading_metrics"]
    
    print(financial_ratios)
    print(financial_ratios['Earnings_Per_Share (EPS)'])
    
    def clean_financials(key_financials):
        cleaned_data = {}
        
        for key, value in key_financials.items():
            try:
                # Remove currency symbols and commas, and convert to float
                str_value = str(value)
                value = str_value.replace('₹', '').replace('â‚¹', '').replace('$', '').replace(',', '').replace('crores', '').replace('crore', '').strip()
                value = value.replace('(TTM)', '').replace('TTM', '').replace('million', '').replace('(estimated)','').strip()
                if value.lower() == 'not provided':
                    cleaned_data[key] = 0.0  # Handle 'Not provided' as 0.0
                else:
                    cleaned_data[key] = float(value)
            except ValueError:
                print(f"Unable to convert value: {value}")
                cleaned_data[key] = 0.0  # Fallback value

        return cleaned_data

    # Clean the financials data for each year
    cleaned_financials = {}
    for year, data in five_year_summary.items():
        cleaned_financials[year] = clean_financials(data)
        
    def clean_earnings_per_share(value):
        if isinstance(value, str):
            cleaned_value = value.replace('â‚¹', '₹').replace('â€”', '—').strip()
            return cleaned_value
        return value  # Return as is if not a string

    # Define the metrics to plot
    metrics = ['sales', 'ebitda', 'net_profit', 'eps', 'total_assets', 'total_current_liabilities', 'LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator', 'total_equity']

    # Set the years for the x-axis
    years = list(five_year_summary.keys())

    # Set up the plot style
    sns.set_theme(style="whitegrid")

    # Define A5 size in inches (approx. 5.8 x 8.3 inches)
    plt.figure(figsize=(5.8, 8.3))

    for metric in metrics:
        # Extract the data for the current metric across all years
        metric_data = [cleaned_financials[year].get(metric, 0.0) for year in years]  # Default to 0.0 if not found

        # Filter out years without valid data
        filtered_years = [years[i] for i in range(len(metric_data)) if metric_data[i] != 0.0]
        filtered_data = [data for data in metric_data if data != 0.0]

        # Plot if there's data to plot
        if filtered_data:
            plt.plot(filtered_years, filtered_data, marker='o', label=metric)

    # Add labels and title
    plt.title(f"5-Year Financial Performance", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Financial Figures (in millions)", fontsize=12)

    # Move the legend to the bottom
    plt.legend(title="Metrics", loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    # Adjust layout to avoid clipping
    plt.tight_layout()

    # Save the plot as an image (e.g., PNG format)
    plt.savefig(f'E:/Datsol/{company_name}_financial_performance.png', bbox_inches='tight')

    # Function to replace currency signs with HTML entities
    def replace_currency_signs(value):
        if isinstance(value, str):
            return value.replace('₹', '&#8377;').replace('$', '&#36;')
        return value

    # Update financial data to replace currency signs
    key_financials = {k: replace_currency_signs(v) for k, v in key_financials.items()}
    five_year_summary = {year: {k: replace_currency_signs(v) for k, v in data.items()} for year, data in five_year_summary.items()}
    # comparable_companies = {k: replace_currency_signs(v) for k, v in comparable_companies.items()}

    html_content = f"""
    <html>
        <head>
            <title>Financial Analysis Report for {company_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    font-size: 20px;
                }}
                h1 {{
                    font-size: 26px;
                    color: #333;
                }}
                h2 {{
                    font-size: 22px;
                    color: #555;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 30px;
                }}
                table, th, td {{
                    border: 1px solid #ddd;
                }}
                th, td {{
                    padding: 12px; /* Increased padding for table cells */
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .sentiment {{
                    font-weight: bold;
                }}
                .horizontal-year th, .horizontal-year td {{
                    text-align: center;
                }}
                @media print {{
                    .page-break {{ page-break-after: always; }}
                    .blank-page {{ height: 100vh; }} 
                }}
            </style>

        </head>
        <body>
            <h1>Financial Analysis Report for {company_name}</h1>
            <h2>About the Company</h2>
            <p>{about_company}</p>
            <h2>Industry Overview</h2>
            <p>{overview}</p>

            <h2>Key Financial Data and Values</h2>
            <table>
                <tr><th>Revenue</th><td>{key_financials['revenue']}</td></tr>
                <tr><th>EBITDA</th><td>{key_financials['ebitda']}</td></tr>
                <tr><th>Net Profit</th><td>{key_financials['net_profit']}</td></tr>
                <tr><th>EPS</th><td>{key_financials['eps']}</td></tr>
                <tr><th>Total Assets</th><td>{key_financials['total_assets']}</td></tr>
                <tr><th>Total Liabilities</th><td>{key_financials['total_current_liabilities']}</td></tr>
                <tr><th>Long Term Debt</th><td>{key_financials['LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator']}</td></tr>
                <tr><th>Total Equity</th><td>{key_financials['total_equity']}</td></tr>
            </table>
            
            

            <h2>5-Year Summary Financials</h2>
            <table class="horizontal-year">
                <thead>
                    <tr>
                        <th>Year</th>
                        {" ".join([f"<th>{year}</th>" for year in five_year_summary.keys()])}
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <th>Sales</th>
                        {" ".join([f"<td>{five_year_summary[year]['sales']}</td>" for year in five_year_summary.keys()])}
                    </tr>
                    <tr>
                        <th>EBITDA</th>
                        {" ".join([f"<td>{five_year_summary[year]['ebitda']}</td>" for year in five_year_summary.keys()])}
                    </tr>
                    <tr>
                        <th>Net Profit</th>
                        {" ".join([f"<td>{five_year_summary[year]['net_profit']}</td>" for year in five_year_summary.keys()])}
                    </tr>
                    <tr>
                        <th>EPS</th>
                        {" ".join([f"<td>{five_year_summary[year]['eps']}</td>" for year in five_year_summary.keys()])}
                    </tr>
                    <tr>
                        <th>Total Assets</th>
                        {" ".join([f"<td>{five_year_summary[year]['total_assets']}</td>" for year in five_year_summary.keys()])}
                    </tr>
                    <tr>
                        <th>Total Liabilities</th>
                        {" ".join([f"<td>{five_year_summary[year]['total_current_liabilities']}</td>" for year in five_year_summary.keys()])}
                    </tr>
                    <tr>
                        <th>Long Term Debt</th>
                        {" ".join([f"<td>{five_year_summary[year]['LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator']}</td>" for year in five_year_summary.keys()])}
                    </tr>
                    <tr>
                        <th>Total Equity</th>
                        {" ".join([f"<td>{five_year_summary[year]['total_equity']}</td>" for year in five_year_summary.keys()])}
                    </tr>
                </tbody>
            </table>

            <h2>Financial Ratios and Analysis</h2>
            <p><strong>CAGR:</strong> {financial_ratios['CAGR']}</p>
            <p><strong>Profit Margin:</strong> {financial_ratios['Profit_Margin']}</p>
            <p><strong>Earnings Per Share (EPS):</strong> {clean_earnings_per_share(financial_ratios['Earnings_Per_Share (EPS)'])}</p>
            <p><strong>Dividend Rate:</strong> {financial_ratios.get('Current_Ratio', 'Not Available')}</p>

            <h2>5-Year Financial Performance</h2>
            <img src="E:/Datsol/{company_name}_financial_performance.png" alt="Financial Performance">
            
            <h2>Investment Sentiment</h2>
            <p class="sentiment"><strong>Recommendation:</strong> {investment_sentiment['recommendation']}</p>
            <p>{investment_sentiment['rationale']}</p>

           

            <h2>Quantitative Summary and Sentiment</h2>
            <p>{quant_summary}</p>
            <p class="sentiment"><strong>Sentiment:</strong> {quant_sentiment}</p>

            <h2>News Summary and Sentiment</h2>
            <p>{news_summary}</p>
            <p class="sentiment"><strong>Sentiment:</strong> {news_sentiment}</p>

        </body>
    </html>
    """
    return html_content

# Function to perform financial analysis and generate HTML output
class StrOutputParser:
    def parse(self, response):
        # Assuming the OpenAI response contains the text in 'choices'
        return response['choices'][0]['message']['content']
    
output_parser = StrOutputParser()

def perform_analysis(org: str):
    news_file = f'{org}_news.csv'
    if os.path.exists(news_file):
        news = pd.read_csv(news_file)
    # Group by date and aggregate both headline and summary into a list
        news['Combined'] = news['headline'] + ": " + news['summary']
        grouped = news.groupby('date')['Combined'].agg(list).reset_index()

        # Function to convert a list of events into a string
        def events_to_string(events):
        # Convert all elements in the list to strings first
            events = [str(event) for event in events]
            
            if len(events) == 1:
                return events[0]  # If only one event, return it directly
            else:
                return ', '.join(events[:-1]) + ' and ' + events[-1]

        # Apply the function to the list of events
        grouped['summary'] = grouped['Combined'].apply(events_to_string)

        # Create the final summary string for each date
        grouped['final_summary'] = grouped.apply(lambda row: f"On {row['date']}, {row['summary']} occurred.", axis=1)

        # Combine all the final summaries
        combined_summary = ' '.join(grouped['final_summary'])

        generation_prompt = f"""
        I would like you to write a summary of the news.
        I am going to tell you some of the relevant article descriptions.
        I will provide you with all the headlines, summaries, and the dates on which they were written.
        Here you go: ```{combined_summary}```
        Write a brief summary of it.
        Disregard all news unrelated to {org}.
        I do not need a summary of each individual day of news.
        Your summary should be about 5 sentences long!
        """
            
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": generation_prompt}],
            model="llama3-8b-8192",
        )
        # Extract the analysis report from the response
        news_summary = chat_completion.choices[0].message.content
        
        sentiment_prompt = f"""
        Identify the sentiment towards the {org} stocks of the news article from -10 to +10 where -10 being the most negative and +10 being the most positive, and 0 being neutral.
        
        GIVE ANSWER IN ONLY ONE WORD AND THAT SHOULD BE THE SCORE. USE POSITIVE OR NEGATIVE SIGN ALONG WITH THE SCORE.

        News summary: {news_summary}
        """

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": sentiment_prompt}],
            model="llama3-8b-8192",
        )

        # Extract the sentiment from the response
        news_sentiment = chat_completion.choices[0].message.content
    else:
            # If the news file is not found, skip the news analysis
            print(f"No news file found for {org}. Skipping news summary and sentiment analysis.")
            news_summary = None
            news_sentiment = None
    
    industry_prompt = f'''Provide an industry overview for the sector that {org} operates in. 
    Focus on the key trends, market growth, challenges, and major players in the industry. 
    Do not provide details about {org} itself, but rather describe the overall industry context. 
    The overview should be about 10 sentences long.'''
  
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": industry_prompt}],
        model="llama3-8b-8192",
    )

    # Extract the sentiment from the response
    industry_overview = chat_completion.choices[0].message.content
    
    annual_results = read_csv_as_string(f'{org}_annual-results.csv')
    annual_prompt = f"""
    Annual results summary: {annual_results}...

    Company name : {org}

    ## Fundamental Analysis Request

    Based on the provided financial summaries for {org}, generate a fundamental analysis report in the following format:

    Generate a fundamental analysis report in the following format:
    5-year summary fundamentals (2019, 2020, 2021, 2022, 2023, 2024):
    Must Use the units specifically for sales, ebitda, net_profit, eps provided in the data((if {org} is indian company use crores if american company use millions))
    "CAGR": {{
            "Total_Revenue": "...",
            "Net_Profit": "...",
            "Operating_Expenses": "...",
        }},
    "Earnings_Per_Share (EPS)": "...",
    "five_year_summary_fundamentals": {{
        "2019": {{
            "sales": "...",
            "ebitda": "...",
            "net_profit": "...",
            "eps": "...",
        }},
        "2020": {{
            "sales": "...",
            "ebitda": "...",
            "net_profit": "...",
            "eps": "...",
        }},
        "2021": {{
            "sales": "...",
            "ebitda": "...",
            "net_profit": "...",
            "eps": "...",
        }},
        "2022": {{
            "sales": "...",
            "ebitda": "...",
            "net_profit": "...",
            "eps": "...",
        }},
        "2023": {{
            "sales": "...",
            "ebitda": "...",
            "net_profit": "...",
            "eps": "...",
        }},
        "2024": {{
            "sales": "...",
            "ebitda": "...",
            "net_profit": "...",
            "eps": "...",
        }}
    }}
    Use numbers wherever required and make sure to be 100 percent confident with your stance. In case you don't find fill Not Provided.
    Don't give any summary just follow the instructed format.
    """

    chat_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "system", "content": annual_prompt}],
        temperature=temperature
    )

    # Parse the response
    annual_report = chat_completion.choices[0]['message']['content']
    
    balance_sheet = read_csv_as_string(f'{org}_balance-sheet.csv')
        # Format the prompt with additional financial metrics
    prompt = f"""
    Annual results summary: {balance_sheet}

    Company name: {org}

    ## Fundamental Analysis Request

    Based on the provided financial summaries for {org}, generate a fundamental analysis report in the following format:

    Generate a fundamental analysis report in the following format:
    5-year summary fundamentals (2019, 2020, 2021, 2022, 2023, 2024)
    Must Use the units specifically for total_assets, total_current_liabilities, LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator, total_equity provided in the data (if {org} is indian company use crores if american company use millions):
    "CAGR": {{
        "Total_Assets": "...",
        "Total_Current_Liabilities": "...",
        "Current_Ratio": "..."
    }},
    "five_year_summary_fundamentals": {{
        "2019": {{
            "total_assets": "...",
            "total_current_liabilities": "...",
            "LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator": "...",
            "total_equity": "..."
        }},
        "2020": {{
            "total_assets": "...",
            "total_current_liabilities": "...",
            "LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator": "...",
            "total_equity": "..."
        }},
        "2021": {{
            "total_assets": "...",
            "total_current_liabilities": "...",
            "LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator": "...",
            "total_equity": "..."
        }},
        "2022": {{
            "total_assets": "...",
            "total_current_liabilities": "...",
            "LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator": "...",
            "total_equity": "..."
        }},
        "2023": {{
            "total_assets": "...",
            "total_current_liabilities": "...",
            "LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator": "...",
            "total_equity": "..."
        }},
        "2024": {{
            "total_assets": "...",
            "total_current_liabilities": "...",
            "LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator": "...",
            "total_equity": "..."
        }}
    }}
    Use numbers wherever required and make sure to be 100 percent confident with your stance. In case you don't find fill Not Provided.
    Don't give any summary just follow the instructed format.
    """

    chat_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "system", "content": prompt}],
        temperature=temperature
    )

    # Parse the response
    balance_report = chat_completion.choices[0]['message']['content']
    print("This is the analysis report:", balance_report)
    
    financial_ratios = read_csv_as_string(f'{org}_financial-ratios.csv')
    prompt = f"""
    financial ratios summary: {financial_ratios}

    Company name: {org}

    ## Fundamental Analysis Request

    Based on the provided financial ratios summary for {org} only fill the CAGR of specified catagories in the following format:
    "Current_Ratio": "...",
    "CAGR": {{
        "ROE_Percent": "...",
        "ROA_Percent": "...",
        "Operating_Expenses": "..."
    }}
    Use numbers wherever required and make sure to be 100 percent confident with your stance.In case you don't find fill Not Provided.
    Don't give any summary just follow the instructed format. 
    """
    
    chat_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "system", "content": prompt}],
        temperature=temperature
    )

    # Parse the response
    cagr_percent = chat_completion.choices[0]['message']['content']
    print("This is the analysis report:", cagr_percent)
    
    cash_flow = read_csv_as_string(f'{org}_cash-flow.csv')
    prompt = f"""
    Cash flow summary: {cash_flow}

    Company name: {org}

    ## Fundamental Analysis Request

    Based on the provided Cash flow summary for {org} only fill the CAGR of specified catagory in the following format:
    "Profit_Margin": "...",
    "Dividend_Rate": "..."
    "CAGR": {{
        "Cash_from_Operating_Activity": "..."
    }}
    Use numbers wherever required and make sure to be 100 percent confident with your stance.In case you don't find fill Not Provided.
    Don't give any summary just follow the instructed format.
    """
    
    chat_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "system", "content": prompt}],
        temperature=temperature
    )

    # Parse the response
    cagr_remaining = chat_completion.choices[0]['message']['content']
    print("This is the analysis report:", cagr_remaining)

    structure_prompt = f'''Combine the {annual_report} , {balance_report}, {cagr_percent} , {cagr_remaining} and must use the units specifically for sales, ebitda, net_profit, eps, total_assets, total_current_liabilities, LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator, total_equity provided in the data structurize according to following format:
{{
    "company_name": "{org}",
    "about_company": "Two-line summary here",
    "key_financial_data_and_values": {{
        "revenue": "...",
        "ebitda": "...",
        "net_profit": "...",
        "eps": "...",
        "total_assets": "...",
        "total_current_liabilities": "...",
        "LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator": "...",
        "total_equity": "..."
    }},
    "financial_ratios_analysis": {{
        "CAGR": {{
            "Total_Revenue": "...",
            "Net_Profit": "...",
            "Operating_Expenses": "...",
            "Total_Assets": "...",
            "Total_Current_Liabilities": "...",
            "ROE_Percent": "...",
            "RoA_Percent": "...",
            "Cash_from_Operating_Activity": "..."
        }},
        "Profit_Margin": "...",
        "Earnings_Per_Share (EPS)": "...",
        "Current_Ratio": "...",
    }},
    "investment_sentiment": {{
        "recommendation": "Strong Buy/Sell",
        "rationale": "..."
    }},
    "five_year_summary_fundamentals": {{
        "2019": {{
            "sales": "...",
            "ebitda": "...",
            "net_profit": "...",
            "eps": "...",
            "total_assets": "...",
            "total_current_liabilities": "...",
            "LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator": "...",
            "total_equity": "..."
        }},
        "2020": {{
            "sales": "...",
            "ebitda": "...",
            "net_profit": "...",
            "eps": "...",
            "total_assets": "...",
            "total_current_liabilities": "...",
            "LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator": "...",
            "total_equity": "..."
        }},
        "2021": {{
            "sales": "...",
            "ebitda": "...",
            "net_profit": "...",
            "eps": "...",
            "total_assets": "...",
            "total_current_liabilities": "...",
            "LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator": "...",
            "total_equity": "..."
        }},
        "2022": {{
            "sales": "...",
            "ebitda": "...",
            "net_profit": "...",
            "eps": "...",
            "total_assets": "...",
            "total_current_liabilities": "...",
            "LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator": "...",
            "total_equity": "..."
        }},
        "2023": {{
            "sales": "...",
            "ebitda": "...",
            "net_profit": "...",
            "eps": "...",
            "total_assets": "...",
            "total_current_liabilities": "...",
            "LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator": "...",
            "total_equity": "..."
        }},
        "2024": {{
            "sales": "...",
            "ebitda": "...",
            "net_profit": "...",
            "eps": "...",
            "total_assets": "...",
            "total_current_liabilities": "...",
            "LongTermLoansAndAdvances_A Long Term Loans Plus Advances Indicator": "...",
            "total_equity": "..."
        }}
    }}
}}
'''


    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "system", "content": structure_prompt}],
        temperature=temperature
    )

    # Parse the response
    structured_response = response.choices[0]['message']['content']
    print(f'''The answer : {structure_prompt}''')
    cleaned_json_string = structured_response.replace('json', '').strip()
    cleaned_string = cleaned_json_string.strip('```')
    content = json.loads(cleaned_string)
    print(type(content))
    print(content)
    
    quant_prompt = f'''I would like you to write a summary on financial report.
    I am going to tell you some of the relevant financial aspects of {org}.
    I will provide you detailed financial reports.
    Here you go: ```{annual_report}, {balance_report},{cagr_percent} and {cagr_remaining}```
    Write a brief summary on it.
    I do not need a summary of each individual line of the report.
    Your summary should be no more than 10 sentences long! '''
    
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "system", "content": quant_prompt}],
        temperature=temperature
    )

    
    quant_summary = response.choices[0]['message']['content']
    
    quant_senti_prompt = f'''Identify the sentiment towards the {org} stocks of the financial report from -10 to +10 where -10 being the most negative and +10 being the most positve , and 0 being neutral

    GIVE ANSWER IN ONLY ONE WORD AND THAT SHOULD BE THE SCORE. USE POSITIVE OR NEGATIVE SIGN ALONG WITH THE SCORE.

    Financial reports : {annual_report}, {balance_report},{cagr_percent} and {cagr_remaining}'''
    
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "system", "content": quant_senti_prompt}],
        temperature=temperature
    )
    
    quant_sentiment = response.choices[0]['message']['content']

    html_report = create_html_report(content,industry_overview,quant_summary,quant_sentiment,news_summary,news_sentiment)
    
    # html_report = html_report.replace("â‚¹", "₹")

    # Save HTML report with correct file name
    html_filename = f"{org}_financial_analysis.html"
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_report)

    # Convert HTML to PDF
    path_to_wkhtmltopdf = r'E:/FinancialBot/wkhtmltopdf/bin/wkhtmltopdf.exe'  # Adjust path if needed
    config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)
    pdf_filename = f"{org}_financial_analysis.pdf"
    
    options = {
    'page-size': 'A5',  # Size of the PDF pages
    'margin-top': '15mm',
    'margin-right': '15mm',
    'margin-bottom': '15mm',
    'margin-left': '15mm',   
    'enable-local-file-access': '',
    }

    pdfkit.from_file(html_filename, pdf_filename, configuration=config, options=options)

    # Verify PDF is generated before download
    if os.path.exists(pdf_filename):
        st.write(f"PDF report saved as {pdf_filename}")

        # Add download button for PDF
        with open(pdf_filename, "rb") as pdf_file:
            st.download_button(
                label="Download PDF Report",
                data=pdf_file,
                file_name=pdf_filename,
                mime="application/pdf"
            )
    else:
        st.error("Failed to generate PDF report.")

# Button to analyze the data
if st.button("Scrape & Analyze"):
    if org_name and url:
        scrape_quantitative_data(org_name, url)
        scrape_news_data(org_name, url)
        perform_analysis(org_name)
    else:
        st.error("Please enter both organization name and URL.")
        
# Add external link under the button
st.markdown(
    f"[DCF Calculator Website]({"https://colabgaurav.github.io/DCF-Calculator/"})",
    unsafe_allow_html=True
)
# Comparable Companies and trading metrics:
#     9. Market Cap : 
#     10. P/E ratio : 
#     11. Price/Book value : 
#     12. Debt to Equity :
#     13. ROCE : 
#     14. Sales Growth 5-year : 
#     15. EBITDA Margin : 
#     16. DCF :

# <h2>Comparable Companies and Trading Metrics</h2>
#             <table>
#                 <tr><th>Market Cap</th><td>{comparable_companies['market_cap']}</td></tr>
#                 <tr><th>P/E Ratio</th><td>{comparable_companies['pe_ratio']}</td></tr>
#                 <tr><th>Price/Book Value</th><td>{comparable_companies['price_book_value']}</td></tr>
#                 <tr><th>Debt to Equity</th><td>{comparable_companies['debt_to_equity']}</td></tr>
#                 <tr><th>ROCE</th><td>{comparable_companies['roce']}</td></tr>
#                 <tr><th>5-Year Sales Growth</th><td>{comparable_companies['sales_growth_5_year']}</td></tr>
#                 <tr><th>EBITDA Margin</th><td>{comparable_companies['ebitda_margin']}</td></tr>
#                 <tr><th>DCF</th><td>{comparable_companies['dcf']}</td></tr>
#             </table>