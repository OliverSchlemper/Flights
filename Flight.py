import pytz
import os
import sqlite3
import re
import uproot
import IPython
import sys
import numpy as np
import pandas as pd
import pymap3d as pm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pandasql import sqldf
from rnog_data.runtable import RunTable
from datetime import datetime, timedelta

from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData

class Flight:
    #------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------
    def __init__(self, flighttracker, i):
        from FlightTracker import FlightTracker
        if not i < len(flighttracker.flights_distinct):
            print(f'index {i} out of bounds for flights_distinct with size {len(flighttracker.flights_distinct)}')
        else:
            self.stations = flighttracker.stations

            self.flightnumber = flighttracker.flights_distinct.flightnumber.iloc[i]
            self.date = flighttracker.flights_distinct.date.iloc[i]

            self.start_time_plot = flighttracker.flights_distinct.mintime.iloc[i][:19] # [:19] to throw away potential microseconds
            self.stop_time_plot = flighttracker.flights_distinct.maxtime.iloc[i][:19] # [:19] to throw away potential microseconds

            self.start_time_plot = FlightTracker.utc.localize(datetime.strptime(self.start_time_plot, FlightTracker.fmt))
            self.stop_time_plot = FlightTracker.utc.localize(datetime.strptime(self.stop_time_plot, FlightTracker.fmt))

            self.header_df = FlightTracker.get_df_from_root_file(self.start_time_plot, self.stop_time_plot)

            self.flights = flighttracker.flights.query( f"readtime_utc >= '{datetime.strftime(self.start_time_plot, FlightTracker.fmt)}' & "
                                                        f"readtime_utc <= '{datetime.strftime(self.stop_time_plot, FlightTracker.fmt)}' & "
                                                        f"flightnumber == '{self.flightnumber}' ").copy()
            self.times = pd.to_datetime(self.flights.readtime_utc, format='ISO8601').astype('int64') / 10**9
            self.r = np.sqrt(self.flights.r2)

            print('''"'''  + self.start_time_plot.strftime("%Y-%m-%dT%H:%M:%S") + '''"''', '''"''' + self.stop_time_plot.strftime("%Y-%m-%dT%H:%M:%S") +  '''"''' + f' duration: {self.stop_time_plot - self.start_time_plot} [hh:mm:ss]')
            print(self.flightnumber)
    #------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------

    def get_what_ever_is_in_that_root_file(start_time, stop_time, filetype = 'headers.root'):
        from FlightTracker import FlightTracker

        if filetype == 'headers.root':
            path = 'header'
        elif filetype == 'combined.root':
            path = 'combined'
        else:
            path = None
            print(f'Unknown file type: {filetype}, choose from ["headers.root", "combined.root"]')

        # getting runtable information and downloading header files
        runtable = FlightTracker.rnogcopy(start_time, stop_time, filetype) 

        # check if files exits for time
        if len(runtable) == 0:
            sys.exit(f'No runs from "{str(start_time)}" to "{str(stop_time)}"')

        #prepare filtering on runtable
        runtable['run_string'] = 'run' + runtable.run.astype(str)
        runtable['station_string'] = 'station' + runtable.station.astype(str)

        # getting filenames for this flight
        filenames = []
        for i in range(len(runtable)):
            try:
                filenames.append([filename for filename in os.listdir(f'./{path}/') if re.search(runtable.station_string.iloc[i], filename) and re.search(runtable.run_string.iloc[i], filename)][0])
            except IndexError:
                print(f'No file with run {runtable.run.iloc[i]} and station {runtable.station.iloc[i]}')
        
        header_df = pd.DataFrame(columns = ['trigger_time', 'station_number', 'radiant_triggers'])
        combined_df = pd.DataFrame(columns = ['trigger_time', 'station_number', 'radiant_triggers'])

        for filename in filenames:
            file = uproot.open(f"{path}/" + filename)
            temp_df = pd.DataFrame(columns = ['trigger_time', 'station_number', 'radiant_triggers'])
            #times = pd.to_datetime(np.array(file['header']['header/trigger_time']), unit = 's')
            # make mast to slice all data
            #mask = (times >= pd.to_datetime(start_time).tz_convert(None)) & (times <= pd.to_datetime(stop_time).tz_convert(None))

            temp_df['trigger_time'] = np.array(file['header']['header/trigger_time'])
            temp_df['station_number'] = np.array(file['header']['header/station_number'])
            temp_df['radiant_triggers'] = np.array(file['header']['header/trigger_info/trigger_info.radiant_trigger'])
            temp_df['lt_triggers'] = np.array(file['header']['header/trigger_info/trigger_info.lt_trigger'])


            if filetype == 'combined.root':
                mask = (times >= start_time) & (times <= stop_time)


        if len(header_df) == 0:
                header_df = temp_df
        else:
            header_df = pd.concat([header_df, temp_df], ignore_index=True, sort=False)


        header_df = header_df[(header_df.trigger_time >= start_time.timestamp()) & (stop_time.timestamp() >= header_df.trigger_time)]
        return header_df



    #------------------------------------------------------------------------------------------------------
    def plot_flight(self):
        from FlightTracker import FlightTracker
        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
        self.fig.subplots_adjust(hspace=0.3, wspace=0.4)
        self.fig.suptitle(self.flightnumber + ', ' + self.date + ''', "''' + str(self.start_time_plot) + '''", "''' + str(self.stop_time_plot) + '''"''')

        #------------------------------------------------------------------------------------------------------
        # ax[0]
        self.ax[0].set_xlabel('longitude [deg]')
        self.ax[0].set_ylabel('latitude [deg]')
        self.ax[0].set_title('trajectory')

        #------------------------------------------------------------------------------------------------------
        # set ticks for time colorbar
        ticks = np.linspace(self.start_time_plot.timestamp(), self.stop_time_plot.timestamp(), 8)
        tick_times = pd.to_datetime(ticks, unit = 's').strftime('%H:%M:%S')
            
        #------------------------------------------------------------------------------------------------------
        # stations
        for i in range(len(self.stations)):
            self.ax[0].scatter(self.stations.longitude[i], self.stations.latitude[i], marker = 'x', label = self.stations['Station Name'][i], s = 15)

        sc = self.ax[0].scatter(self.flights.longitude, self.flights.latitude, marker = '.', c = self.times, cmap = 'viridis', s = 15)
        cbar = self.fig.colorbar(sc, ax=self.ax[0])
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_times)

        self.ax[0].legend()

        #------------------------------------------------------------------------------------------------------
        # ax[1]
        self.n_bins = np.arange(self.start_time_plot.timestamp(), self.stop_time_plot.timestamp(), 10)
        n_bins = self.n_bins

        self.ax[1].hist(self.header_df[self.header_df.lt_triggers == True].trigger_time, bins = n_bins, color = 'C0',  label = 'lt triggers', histtype = 'step', linewidth = 2, alpha = 0.5)
        self.ax[1].hist(self.header_df[self.header_df.radiant_triggers == True].trigger_time, bins = n_bins, color = 'C1',  label = 'radiant triggers', histtype = 'step', linewidth = 2, alpha = 0.5)


        self.ax_01_twin = self.ax[1].twinx()
        self.ax_01_twin.plot(self.times, self.r, '.', markersize = 3, label = 'd [km]', color = 'C4')

        x = np.linspace(self.start_time_plot.timestamp(), self.stop_time_plot.timestamp(), 100)
        #self.ax_01_twin.plot(x[1:-1], FlightTracker.part_lin(x[1:-1], times, r), '-')
        #ax_01_twin.plot(times, f.altitude/1000, 'x', color = 'C5')
        self.ax_01_twin.plot(self.times, self.flights.z, '.', markersize = 3, label = 'altitude [km]', color = 'C6')
        #ax_01_twin.plot(times, np.sqrt(f.x**2 + f.y**2 + f.z**2), 'x', color = 'C7')

        self.ax[1].set_title('Sum all stations')
        self.ax[1].set_xticks(ticks)
        self.ax[1].set_xticklabels(tick_times, rotation=90)
        self.ax[1].set_xlim(min(ticks), max(ticks))
        self.ax[1].legend()
        self.ax_01_twin.legend(loc = 1)

