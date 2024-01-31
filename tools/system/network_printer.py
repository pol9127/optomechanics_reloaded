import requests
from bs4 import BeautifulSoup
from datetime import datetime as dt

class Printer:
    _properties = {}
    url = None
    def __init__(self):
        pass

    @property
    def _query_properties(self):
        return None, None

    @property
    def properties(self):
        response = requests.get(self.url, verify=False)
        self.soup = BeautifulSoup(response.text, 'html.parser')
        colors, properties, capacities = self._query_properties
        color_information = {col: prop for col, prop in zip(colors, properties)}
        for cap, col in zip(capacities, colors):
            color_information[col]['capacity'] = cap
        return color_information


def custom_dt(dt_string):
    cust_dt = dt.strptime(dt_string, '%Y%m%d')
    return cust_dt


class HPLaserJetMFP(Printer):
    type = 'M476dw'
    url = 'http://129.132.1.138/info_suppliesStatus.html?tab=Home&menu=SupplyStatus'
    dtypes = [str, int, int, str, custom_dt, custom_dt]

    @property
    def _query_properties(self):
        color_information = self.soup.findAll('table')[6:16:3]
        colors = [col_info.findAll('td')[2].text.strip().split('\n')[0] for col_info in color_information]
        capacities = [col_info.findAll('td')[3].text.strip().split('\n')[0].replace('--', '0') for col_info in color_information]

        props = [9, 11, 13, 15, 17, 19]
        vals = [10, 12, 14, 16, 18, 20]
        properties = [{col_info.findAll('td')[prop].text.strip().split('\n')[0].replace(':', ''):
                           dtyp(col_info.findAll('td')[val].text.replace('>', '').strip().split('\n')[0].replace('--', '0'))
                       for prop, val, dtyp in zip(props, vals, self.dtypes)}
                      for col_info in color_information]
        return colors, properties, capacities


class HPLaserJetP3015(Printer):
    type = 'P3015'
    url = 'https://129.132.1.144/hp/device/this.LCDispatcher?nav=hp.Supplies'
    dtypes = [int, str, int, int, custom_dt, custom_dt]
    @property
    def _query_properties(self):
        colors = [self.soup.findAll('span')[0].text.strip().split('\n')[0]]
        capacities = [self.soup.findAll('span')[2].text.strip().split('\n')[0].replace('--', '0').replace('*', '')]
        items = ['itm-9429', 'itm-4367', 'itm-9681', 'itm-9079', 'itm-9695', 'itm-9697']
        properties = [{self.soup.find(id=item).findAll('div')[0].text.strip().replace(':', '').replace('*', ''):
                       dtyp(self.soup.find(id=item).findAll('div')[1].text.replace('>', '').strip().replace('--', '0'))
                       for item, dtyp in zip(items, self.dtypes)}]
        return colors, properties, capacities


if __name__ == '__main__':
    color_printer = HPLaserJetMFP()
    bw_printer = HPLaserJetP3015()
    print(bw_printer.properties)
    print(color_printer.properties)
