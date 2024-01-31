from PyQt5 import uic, QtCore, QtWidgets, QtGui
import sys
import json
import feedparser
import time
import re
import urllib.request
import os.path
from os.path import isfile
relative = True
if relative:
    base_path = ''
else:
    base_path = sys.path[0][:-16]

# d = feedparser.parse('http://feeds.feedburner.com/acs/apchd5')

class GUI(QtWidgets.QMainWindow):
    html_rep = {'<br/>' : '\n'}
    html_rep = dict((re.escape(k), v) for k, v in html_rep.items())
    html_pattern = re.compile("|".join(html_rep.keys()))
    feeds = {}
    parsers = {}
    ids = {}
    feed_table_header = ['Title', 'Author', 'Published', 'ID', 'Journal']
    xml_property = {'Title': ['title'], 'Author': ['author', 'authors'],
                    'Published': ['published_parsed', 'published'], 'ID' : ['id']}
    detail_key_order = ['title', ['author', 'authors'], 'published', 'summary', ['link', 'id']]
    def __init__(self, **kwargs):
        QtWidgets.QDialog.__init__(self)
        self.ui = uic.loadUi(base_path + 'mainWindow.ui')
        self.ui.showMaximized()
        self.load_favorite_feeds()
        self.ui.pushButton_saveFavorites.clicked.connect(self.save_favorite_feeds)
        self.ui.tableWidget_settings.cellChanged.connect(self.update_tableWidget_settings)
        self.ui.pushButton_updateFeeds.clicked.connect(self.update_feeds)
        self.ui.pushButton_refresh.clicked.connect(self.refresh_feed_table)
        self.ui.tableWidget_feeds.cellClicked.connect(lambda row, col: self.get_entry(row))

        self.ui.tableWidget_feeds.setColumnCount(len(self.feed_table_header))
        self.ui.tableWidget_feeds.setRowCount(1)
        for col, head in enumerate(self.feed_table_header):
            item = QtWidgets.QTableWidgetItem(head)
            self.ui.tableWidget_feeds.setHorizontalHeaderItem(col, item)
        self.ui.tableWidget_feeds.setColumnHidden(len(self.feed_table_header) - 2, True)


    def load_favorite_feeds(self):
        if isfile(base_path + 'feeds.json'):
            with open(base_path + 'feeds.json', 'r') as jsonfile:
                self.feeds = json.load(jsonfile)
            keys = list(self.feeds.keys())
            self.ui.tableWidget_settings.setRowCount(len(keys) + 1)
            for row, key in enumerate(keys):
                item = QtWidgets.QTableWidgetItem(key)
                self.ui.tableWidget_settings.setItem(row, 0, item)
                item = QtWidgets.QTableWidgetItem(self.feeds[key])
                self.ui.tableWidget_settings.setItem(row, 1, item)
            for feed in self.feeds:
                self.parsers[feed] = feedparser.parse(self.feeds[feed])
                self.ids[feed] = ['{0}_{1}'.format(feed, i) for i in range(len(self.parsers[feed].entries))]
        else:
            print('No favorite feeds found.')
        # print(list(self.parsers['ACS Photonics'].entries[0].keys()))

        # print(list(self.parsers['IEEE Journal of Quantum Electronics'].entries[0].keys()))
        # print(list(self.parsers['Optics Express'].entries[0].keys()))
        # html_decoded = self.replace_html(self.parsers['Optics Express'].entries[0]['summary'])
        # print(self.parsers['Optics Express'].entries[0]['summary'])
        # print(html_decoded)
        # print(item_text)

    def save_favorite_feeds(self):
        self.update_feeds()
        with open(base_path + 'feeds.json', 'w') as jsonfile:
            json.dump(self.feeds, jsonfile)

    def get_table(self, table):
        content = []
        for row in range(table.rowCount()):
            content_row = []
            for col in range(table.columnCount()):
                item = table.item(row, col)
                if item is not None:
                    content_row.append(table.item(row, col).text())
                else:
                    content_row.append('')
            content.append(content_row)
        return content

    def update_feeds(self):
        self.feeds = {}
        self.parsers = {}
        self.ids = {}
        table = self.get_table(self.ui.tableWidget_settings)
        for row in table:
            if row[0] != '' and row[1] != '':
                self.feeds[row[0]] = row[1]

        for feed in self.feeds:
            self.parsers[feed] = feedparser.parse(self.feeds[feed])
            self.ids[feed] = ['{0}_{1}'.format(feed, i) for i in range(len(self.parsers[feed].entries))]

    def update_tableWidget_settings(self):
        table = self.get_table(self.ui.tableWidget_settings)
        for n, row in enumerate(table[::-1]):
            if row[0] != '' or row[1] != '':
                self.ui.tableWidget_settings.setRowCount(self.ui.tableWidget_settings.rowCount() - n + 1)
                return
        self.ui.tableWidget_settings.setRowCount(1)

    def refresh_feed_table(self):
        self.ui.tableWidget_feeds.clearContents()
        rowCount = sum([len(self.parsers[pars].entries) for pars in self.parsers])
        self.ui.tableWidget_feeds.setRowCount(rowCount)
        n = 0
        for pars in self.parsers:
            for entry, id_ in zip(self.parsers[pars].entries, self.ids[pars]):
                journal = QtWidgets.QTableWidgetItem(pars)
                journal.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                self.ui.tableWidget_feeds.setItem(n, len(self.feed_table_header) - 1, journal)
                id_item = QtWidgets.QTableWidgetItem(id_)
                self.ui.tableWidget_feeds.setItem(n, len(self.feed_table_header) - 2, id_item)
                for col, header in enumerate(self.feed_table_header[:-2]):
                    item = QtWidgets.QTableWidgetItem(self.get_property(entry, self.xml_property[header]))
                    item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                    self.ui.tableWidget_feeds.setItem(n, col, item)
                n += 1
        self.ui.tableWidget_feeds.setColumnWidth(0, 300)

    def get_property(self, entry, property):
        for prop in property:
            if prop in entry:
                val = entry[prop]
                if val is not None:
                    if isinstance(val, time.struct_time):
                        val = time.strftime('%Y-%m-%d', val)
                    return val
        return ''

    def get_entry(self, row):
        id_ = self.ui.tableWidget_feeds.item(row, len(self.feed_table_header) - 2)
        if id_ is not None:
            feed, num = id_.text().rsplit('_', 1)
            num = int(num)
            entry = self.parsers[feed].entries[num]
            self.populate_details(entry)

    def populate_details(self, entry):
        self.ui.tableWidget_details.clearContents()
        keys_unordered = [k for k in entry if not '_parsed' in k]
        keys = []
        for desired_key in self.detail_key_order:
            if isinstance(desired_key, str):
                if desired_key in keys_unordered:
                    keys.append(desired_key)
            elif isinstance(desired_key, list):
                for d_k in desired_key:
                    if d_k in keys_unordered:
                        keys.append(d_k)
                        break

        for key in keys_unordered:
            if key not in keys:
                keys.append(key)
        values = [entry[k] for k in keys]
        self.ui.tableWidget_details.setRowCount(len(keys))
        self.ui.tableWidget_details.setColumnCount(1)

        for n, (k, v) in enumerate(zip(keys, values)):
            header_item = QtWidgets.QTableWidgetItem(k)
            if isinstance(v, str):
                item_text = self.replace_html(v)
                img_loc = self.find_image_link_html(item_text)
                item_text = self.remove_html(item_text)
                self.ui.tableWidget_details.setVerticalHeaderItem(n, header_item)
                if img_loc is not None:
                    icon = QtGui.QPixmap(img_loc)
                    item = QtWidgets.QTableWidgetItem()
                    item_txt = QtWidgets.QTableWidgetItem(item_text)
                    item.setData(1, icon)
                    subtable = self.create_subtable(n)
                    subtable.setRowCount(1)
                    subtable.setColumnCount(2)
                    subtable.verticalHeader().hide()
                    subtable.setItem(0, 0, item)
                    subtable.setItem(0, 1, item_txt)
                    subtable.resizeRowsToContents()
                    subtable.resizeColumnsToContents()
                else:
                    item = QtWidgets.QTableWidgetItem(item_text)
                    self.ui.tableWidget_details.setItem(n, 0, item)
            elif isinstance(v, feedparser.FeedParserDict):
                self.ui.tableWidget_details.setVerticalHeaderItem(n, header_item)
                subtable = self.create_subtable(n)
                keys_sub = [k_sub for k_sub in v if not '_parsed' in k_sub]
                values_sub = [v[k_sub] for k_sub in keys_sub]
                subtable.setRowCount(len(keys_sub))
                subtable.setColumnCount(1)
                for n_sub, (k_sub, v_sub) in enumerate(zip(keys_sub, values_sub)):
                    header_item_sub = QtWidgets.QTableWidgetItem(k_sub)
                    if isinstance(v_sub, str):
                        item_text_sub = self.replace_html(v_sub)
                        img_loc = self.find_image_link_html(item_text_sub)
                        item_text_sub = self.remove_html(item_text_sub)
                        subtable.setVerticalHeaderItem(n_sub, header_item_sub)
                        if img_loc is not None:
                            icon = QtGui.QPixmap(img_loc)
                            item = QtWidgets.QTableWidgetItem()
                            item_txt_sub = QtWidgets.QTableWidgetItem(item_text_sub)
                            item.setData(1, icon)
                            subtable_sub = self.create_subtable(n_sub, subtable)
                            subtable_sub.setRowCount(1)
                            subtable_sub.setColumnCount(2)
                            subtable_sub.verticalHeader().hide()
                            subtable_sub.setItem(0, 0, item)
                            subtable_sub.setItem(0, 1, item_txt_sub)
                            subtable_sub.resizeRowsToContents()
                            subtable_sub.resizeColumnsToContents()
                        else:
                            item_sub = QtWidgets.QTableWidgetItem(item_text_sub)
                            subtable.setItem(n_sub, 0, item_sub)
                    else:
                        item_sub = QtWidgets.QTableWidgetItem(str(v_sub))
                        subtable.setVerticalHeaderItem(n_sub, header_item_sub)
                        subtable.setItem(n_sub, 0, item_sub)
            elif isinstance(v, list):
                self.ui.tableWidget_details.setVerticalHeaderItem(n, header_item)
                subtable = self.create_subtable(n)
                v = [l for l in v if isinstance(l, feedparser.FeedParserDict)]
                rowCount = sum([len(list(l.keys())) for l in v])
                subtable.setRowCount(rowCount)
                subtable.setColumnCount(1)
                n_sub = 0
                for l in v:
                    keys_sub = [k_sub for k_sub in l if not '_parsed' in k_sub]
                    values_sub = [l[k_sub] for k_sub in keys_sub]
                    for k_sub, v_sub in zip(keys_sub, values_sub):
                        header_item_sub = QtWidgets.QTableWidgetItem(k_sub)
                        if isinstance(v_sub, str):
                            item_text_sub = self.replace_html(v_sub)
                            img_loc = self.find_image_link_html(item_text_sub)
                            item_text_sub = self.remove_html(item_text_sub)
                            subtable.setVerticalHeaderItem(n_sub, header_item_sub)
                            if img_loc is not None:
                                icon = QtGui.QPixmap(img_loc)
                                item = QtWidgets.QTableWidgetItem()
                                item_txt_sub = QtWidgets.QTableWidgetItem(item_text_sub)
                                item.setData(1, icon)
                                subtable_sub = self.create_subtable(n_sub, subtable)
                                subtable_sub.setRowCount(1)
                                subtable_sub.setColumnCount(2)
                                subtable_sub.verticalHeader().hide()
                                subtable_sub.setItem(0, 0, item)
                                subtable_sub.setItem(0, 1, item_txt_sub)
                                subtable_sub.resizeRowsToContents()
                                subtable_sub.resizeColumnsToContents()
                            else:
                                item_sub = QtWidgets.QTableWidgetItem(item_text_sub)
                                subtable.setItem(n_sub, 0, item_sub)
                        else:
                            item_sub = QtWidgets.QTableWidgetItem(str(v_sub))
                            subtable.setVerticalHeaderItem(n_sub, header_item_sub)
                            subtable.setItem(n_sub, 0, item_sub)
                        n_sub += 1

            self.ui.tableWidget_details.resizeRowsToContents()


    def create_subtable(self, row, parent_table=None):
        if parent_table is None:
            parent_table = self.ui.tableWidget_details
        table = QtWidgets.QTableWidget()
        table.horizontalHeader().hide()
        table.horizontalHeader().setStretchLastSection(True)
        parent_table.setCellWidget(row, 0, table)
        return table

    def replace_html(self, html_str):
        return self.html_pattern.sub(lambda m: self.html_rep[re.escape(m.group(0))], html_str)

    def find_image_link_html(self, html_str):
        image_types = ['.gif', '.jpg', '.png']
        for typ in image_types:
            result = re.search('http(.*){0}'.format(typ), html_str)
            if result is not None:
                image_link = 'http{0}{1}'.format(result.group(1), typ)
                image_name = image_link.rsplit('/', 1)[-1]
                image_path = os.path.join('tmp', image_name)
                if not os.path.isfile(image_path):
                    urllib.request.urlretrieve(image_link, os.path.join('tmp', image_name))
                return image_path


    def remove_html(self, html_str):
        return re.sub('<[^>]+>', '', html_str)

    def exitHandler(self):
        for f in os.listdir('tmp'):
            os.remove(os.path.join('tmp', f))

class ImageWidget(QtWidgets.QWidget):
    def __init__(self, imagePath, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.pic = QtGui.QPixmap(imagePath)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawPixmap(0, 0, self.pic)
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = GUI()
    app.aboutToQuit.connect(window.exitHandler)
    sys.exit(app.exec_())
