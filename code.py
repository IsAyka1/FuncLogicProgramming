from bs4 import BeautifulSoup 
import requests as req
 
resp = req.get("https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv")
soup = BeautifulSoup(resp.text, 'lxml')
root = soup.body
for tag in soup.find_all("td", {"class" :"blob-code blob-code-inner js-file-line"}):
        print("{0}: {1}".format(tag.name, tag.text))