import yfinance as yf

#proxy_dict = {"http": "http://172.31.100.25:3128", "https": "http://172.31.100.25:3128"}
        
yf_search = yf.Search("Market", news_count=15)

print(yf_search.news)

