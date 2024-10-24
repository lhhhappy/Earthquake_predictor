from datetime import datetime
import pandas as pd
from libcomcat.search import search
import os
os.makedirs('usgs_data', exist_ok=True)
for year in range(1986, 2024):
    box_events = search(starttime=datetime(year, 1, 1, 00, 00), endtime=datetime(year+1, 1, 1, 00, 00),
                   minlatitude=32, maxlatitude=36, minlongitude=-120, maxlongitude=-114)
    events_data = [{
    'ID': event.id,
    'Time': event.time,
    'Magnitude': event.magnitude,
    'Latitude': event.latitude,
    'Longitude': event.longitude,
    'Depth': event.depth
    } for event in box_events]
    df = pd.DataFrame(events_data)
    df.to_csv('usgs_data/earthquakes_{}.csv'.format(year), index=False)
    print('Year {} has {} events'.format(year, len(events_data)))
    