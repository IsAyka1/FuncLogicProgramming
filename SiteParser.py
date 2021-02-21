from bs4 import BeautifulSoup
import requests as req
import random


def site_parser():
    # resp = req.get("https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv")
    resp = req.get("https://github.com/IsAyka1/FuncLogicProgramming/blob/master/NewData.txt")
    soup = BeautifulSoup(resp.text, 'lxml')
    root = soup.body
    outputFileName = 'RandomData.txt'
    with open(outputFileName, 'w') as file:
        list = []
        for tag in soup.find_all("td", {"class": "blob-code blob-code-inner js-file-line"}):
	    list.append(tag.text)
        file.write(random.shuffle(list))
    return outputFileName


if __name__ == '__main__':
    site_parser()
