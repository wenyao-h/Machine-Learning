# -*- coding: utf-8 -*-
"""
shareholder information of a stock are listed in :
https://q.stock.sohu.com/cn/000001/ltgd.shtml
https://q.stock.sohu.com/cn/000002/ltgd.shtml
https://q.stock.sohu.com/cn/000003/ltgd.shtml
...

And you are requried to collect the tables of shareholder information for stocks in "sz50_top10_stocks"
with following 7 columns, and then perform the analysis to answer the questions.
    1. 'stock'-股票代码
    2. 'rank'-排名
    3. 'org_name'-股东名称	
    4. 'shares'-持股数量(万股)
    5. 'percentage'-持股比例	
    6. 'changes'-持股变化(万股)
    7. 'nature'-股本性质
    
Please pay attention to the data types of different columns, especially 'rank', 'percentage'

"""

from pickletools import float8
from unicodedata import numeric
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
fake_header = {  "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36",
            "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding":"gzip, deflate, sdch",
            "Accept-Language":"zh-TW,zh;q=0.8,en-US;q=0.6,en;q=0.4,zh-CN;q=0.2"
        }

data_file= './data/stock_shareholders.csv'
sz50_stocks=('600000','600958','601985','600029','600111','601857','601818',
             '601800','601788','601766','601688','601668','601628','601601',
             '601398','601390','601336','601328','601318','601288','601229',
             '601211','601198','601186','601169','601166','601088','601006',
             '600999','601988','600919','600887','600837','600606','600547',
             '600519','600518','600485','600340','601881','600104','600100',
             '600050','600048','600036','600030','601901','600028','600016',
             '601989')

sz50_top10_stocks = sz50_stocks[:10]

print('There are',len(sz50_top10_stocks), 'stocks in sz50_top10_stocks')

base_url = 'https://q.stock.sohu.com/cn/{}/ltgd.shtml' 
row_count = 0
#create a list to store the crawled share-holdoing records
results=[]
for stock in sz50_top10_stocks:#process stock one by one
    #prepare the request webpage with desired parameters
    url = base_url.format(stock)
    print("Now we are crawling stock",stock)
    #send http request with fake http header
    response = requests.get(url,headers = fake_header)
    if response.status_code == 200:
        #++insert your code here++ to set the encoding of response according to the charset of html
        response.encoding = 'gb2312'
        #源代码开头的标签<meta http-equiv="Content-Type" content="text/html; charset=gb2312" />
        
        root = BeautifulSoup(response.text,"html.parser") 
        # search the table storing the shareholder information
        table = root.find_all('table',class_='tableG') #++insert your code here++
        # list all rows the table, i.e., tr tags
        rows = table[0].find_all('tr')
        #注意用find_all后会返回一个resultset，而这个object中的元素是tag，才可以继续用find系列公式

        for row in rows: #iterate rows
            record=[stock,]# define a record with stock pre-filled and then store columns of the row/record
            # list all columns of the row , i.e., td tags
            columns = row.find_all('td') #++insert your code here++
            for col in columns: #iterate columns
                record.append(col.get_text().strip())
            if len(record) == 7:# if has valid columns, save the record to list results
                results.append(record)
                row_count+=1
        time.sleep(1)
print('Crawled and saved {} records of  shareholder information of sz50_top10_stocks to{}'.format(row_count,data_file) )

sharehold_records_df = pd.DataFrame(columns=['stock', 'rank','org_name','shares','percentage','changes','nature'], data=results)

sharehold_records_df.to_excel("./data/sharehold_records.xlsx")


print("List of shareholers are \n", sharehold_records_df['org_name'])


#insert your code here for Q1-1, Q1-2 and Q1-3

#Q1-1 How many unique organizations that are among the top 5 ('rank'<=5) shareholders of any stocks in  sz50_top10
sharehold_records_df[sharehold_records_df['rank'].astype(int)<=5]['org_name'].value_counts().count()

#Q1-2 How many organization (full name) holds more than 3 different stocks （with any rank）  in sz50_top10?
res = sharehold_records_df[['stock','org_name']].groupby('org_name').count()
res[res['stock']>3].count()

#Q1-3 Which organization (full name) holds the most total percentage of shares among all stocks in is  sz50_top10?
new_df = sharehold_records_df
new_df['percentage'] = new_df.loc[:,'percentage'].str.rstrip('%').astype('float') / 100.0

res = new_df[['org_name','percentage']].groupby('org_name').sum().sort_values(by="percentage")
res.iloc[-1]
