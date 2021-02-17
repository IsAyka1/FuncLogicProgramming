from bs4 import BeautifulSoup
import requests as req


def site_parser():
    # resp = req.get("https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv")
    # soup = BeautifulSoup(resp.text, 'lxml')
    # root = soup.body
    # outputFileName = 'siteParserOutput.txt'
    # with open(outputFileName, 'w') as file:
    #     for tag in soup.find_all("td", {"class": "blob-code blob-code-inner js-file-line"}):
    #         file.write('{data}\n'.format(data=tag.text))
    #  need shuffle all data!!!
    outputFileName = 'NewData.txt'
    return outputFileName


if __name__ == '__main__':
    site_parser()
