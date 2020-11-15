import requests
from bs4 import BeautifulSoup


main_url = "http://textfiles.com/sex/EROTICA"
r  = requests.get(main_url)
data = r.text
soup = BeautifulSoup(data)
num = 1
for link in soup.find_all('a'):
    second_url = link.get('href')
    new_url = main_url+'/'+second_url+'/'
    new_r = requests.get(new_url)
    new_data = new_r.text
    new_soup = BeautifulSoup(new_data)
    for new_link in new_soup.find_all('a'):
        # print(new_url+new_link.get('href'))
        if num>=4702:
            response = new_url+new_link.get('href')
            cont = requests.get(response, allow_redirects=True)
            open(f'file_{num}.txt','wb').write(cont.content)
        num+=1
print(num)