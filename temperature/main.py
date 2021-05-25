import pandas as pd
import numpy as np
import time

from bokeh.plotting import figure, output_file, show
from bokeh.tile_providers import STAMEN_TONER, get_provider
from bokeh.palettes import OrRd
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.transform import linear_cmap

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

def coord(x, y):
    lat = x
    lon = y
    r_major = 6378137.000
    x = r_major * np.radians(lon)
    scale = x/lon
    y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + lat * (np.pi/180.0)/2.0)) * scale
    return (x, y)

index = ["id", "nome", "latitude", "longitude", "capital", "codigo"]
town = pd.read_csv("municipios.csv", names=index)

plot_data = pd.DataFrame(columns=["t", "mercator_x", "mercator_y"])
timeout = 10

try:
    for i in range(1, 10):
        name = town.iloc[i]["nome"]
        lat = float(town.iloc[i]["latitude"])
        lon = float(town.iloc[i]["longitude"])

        try:
            driver = webdriver.Firefox()
            driver.get("https://www.climatempo.com.br/")
            input = driver.find_element_by_id("searchGeneral")
            input.send_keys(name)

            time.sleep(2)
            city_autocomplete = driver.find_element_by_id("searchGeneral_autocomplete")
            li_item = city_autocomplete.find_elements_by_tag_name("li")[1]
            link = li_item.find_element_by_tag_name("a")
            link.click()

            element_present = EC.presence_of_element_located((By.ID, 'min-temp-1'))
            WebDriverWait(driver, timeout).until(element_present)
            min_t = int(driver.find_element_by_id("min-temp-1").text.strip("°"))
            max_t = int(driver.find_element_by_id("max-temp-1").text.strip("°"))

            c = coord(lat, lon)
            plot_data = plot_data.append({"t": (min_t + max_t) / 2, "mercator_x": c[0], "mercator_y": c[1]}, ignore_index=True)
        except Exception as e:
            print("Error: " + name)

        driver.close()
except KeyboardInterrupt:
    pass

source = ColumnDataSource(plot_data)
color_mapper = linear_cmap(field_name = 't', palette=OrRd[9], low=plot_data['t'].min(), high=plot_data['t'].max())
color_bar = ColorBar(color_mapper=color_mapper['transform'], formatter=NumeralTickFormatter(format='0.0[0000]'),
                     label_standoff=13, width=8, location=(0,0))

output_file("map.html")
tooltips=[("Temperatura", "@t")]
p = figure(title="Temperatura Brasil", x_range=(-2000000, 6000000), y_range=(-1000000, 7000000),
           x_axis_type="mercator", y_axis_type="mercator", x_axis_label="Longitude", y_axis_label="Latitude", tooltips=tooltips)
p.add_tile(get_provider(STAMEN_TONER))
p.circle(x='mercator_x', y='mercator_y', color=color_mapper, source=source, size=15, fill_alpha=0.7)
p.add_layout(color_bar, 'right')

show(p)
