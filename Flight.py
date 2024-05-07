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
from NuRadioReco.utilities import units

from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData

class Flight:

    path_to_combined_files = '/home/oliver/software/Flights/combined/'
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

            self.header_df = Flight.get_what_ever_is_in_those_root_files(self.start_time_plot, self.stop_time_plot)

            self.flights = flighttracker.flights.query( f"readtime_utc >= '{datetime.strftime(self.start_time_plot, FlightTracker.fmt)}' & "
                                                        f"readtime_utc <= '{datetime.strftime(self.stop_time_plot, FlightTracker.fmt)}' & "
                                                        f"flightnumber == '{self.flightnumber}' ").copy()
            self.times = pd.to_datetime(self.flights.readtime_utc, format='ISO8601').astype('int64') / 10**9
            self.r = np.sqrt(self.flights.r2)

            print('''"'''  + self.start_time_plot.strftime("%Y-%m-%dT%H:%M:%S") + '''"''', '''"''' + self.stop_time_plot.strftime("%Y-%m-%dT%H:%M:%S") +  '''"''' + f' duration: {self.stop_time_plot - self.start_time_plot} [hh:mm:ss]')
            print(self.flightnumber)
    #------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------------
    def calc_l1_max_and_amp_max_and_SNR_max(event, station_number):
        #print(station_number)
        l1_max = 0
        amp_max = 0
        SNR_max = 0
        station = event.get_station(station_number)
        for i in range(24):
            channel = station.get_channel(i)
            trace = np.abs(channel.get_trace())
            #times = channel.get_times()

            freq = channel.get_frequencies()
            mask = (0.05 < freq) & (freq < 0.8) & (freq != 0.2)
            freq = freq[mask]
            spectrum = np.abs(channel.get_frequency_spectrum())[mask]

            #calculate
            l1 = Flight.simple_l1(spectrum)
            amp = np.max(trace)
            avg = np.average(trace)
            SNR = amp / avg

            #check
            if l1 > l1_max:
                l1_max = l1
            if amp > amp_max:
                amp_max = amp
            if SNR > SNR_max:
                SNR_max = SNR
        return l1_max, amp_max, SNR_max

    #------------------------------------------------------------------------------------------------------
    
    def simple_l1(frequencies):
        return np.max(frequencies**2)/np.sum(frequencies**2)
    #------------------------------------------------------------------------------------------------------
    def get_what_ever_is_in_those_root_files(start_time, stop_time, filetype = 'headers.root'):
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

        for filename in filenames:
            file = uproot.open(f"{path}/" + filename)
            temp_df = pd.DataFrame()
            
            '''
            # make mask to slice all data
            times = pd.to_datetime(np.array(file[path]['header/trigger_time']), unit = 's')
            mask = (times >= pd.to_datetime(start_time).tz_convert(None)) & (times <= pd.to_datetime(stop_time).tz_convert(None))

            # header information
            temp_df['station_number'] = np.array(file[path]['header/station_number'])[mask]
            temp_df['run_number'] = np.array(file[path]['header/run_number'])[mask]
            temp_df['event_number'] = np.array(file[path]['header/event_number'])[mask]
            temp_df['trigger_time'] = np.array(file[path]['header/trigger_time'])[mask]
            temp_df['radiant_triggers'] = np.array(file[path]['header/trigger_info/trigger_info.radiant_trigger'])[mask]
            temp_df['lt_triggers'] = np.array(file[path]['header/trigger_info/trigger_info.lt_trigger'])[mask]
            run_nr = np.array(file[path]['header/run_number'])[0]
            # combinded (waveform) information
            '''

#            header information
            temp_df['station_number'] = np.array(file[path]['header/station_number'])
            temp_df['run_number'] = np.array(file[path]['header/run_number'])
            temp_df['event_number'] = np.array(file[path]['header/event_number'])
            temp_df['trigger_time'] = np.array(file[path]['header/trigger_time'])
            temp_df['radiant_triggers'] = np.array(file[path]['header/trigger_info/trigger_info.radiant_trigger'])
            temp_df['lt_triggers'] = np.array(file[path]['header/trigger_info/trigger_info.lt_trigger'])
            run_nr = np.array(file[path]['header/run_number'])[0]
            # combinded (waveform) information

            if filetype == 'combined.root':
                
                # if combined scores already exist for that root file (run, station) then join them instead of calculating
                path_combined_scores = f'./combined_scores/{filename[:-5]}_scores.db' # remove '.root' from filename
                if os.path.exists(path_combined_scores): 
                    # Establish a connection to the SQLite database
                    con = sqlite3.connect(path_combined_scores)
                    
                    # get combined_scores from db file and join on temp_df
                    temp_scores = pd.read_sql_query("SELECT * FROM combined_scores", con)
                    temp_df = temp_df.merge(temp_scores, on=['station_number', 'run_number', 'event_number'], how='left')
                    
                    # Close the database connection
                    con.close()
                else:
                    reader = readRNOGData()

                    reader.begin([f'{Flight.path_to_combined_files}{filename}'], overwrite_sampling_rate=3200*units.MHz)

                    l1s = np.zeros(len(temp_df.event_number))
                    amps = np.zeros(len(temp_df.event_number))
                    SNRs = np.zeros(len(temp_df.event_number))
                    for i in range(len(temp_df.event_number)):
                        l1, amp, SNR = Flight.calc_l1_max_and_amp_max_and_SNR_max(reader.get_event(run_nr=run_nr, event_id=temp_df.event_number.iloc[i]), temp_df.station_number.iloc[i])
                        l1s[i] = l1
                        amps[i] = amp
                        SNRs[i] = SNR

                    temp_df['l1_max'] = l1s
                    temp_df['amp_max'] = amps
                    temp_df['SNR_max'] = SNRs
                    l1_threshold = 0.4
                    SNR_threshold = 10
                    temp_df['cw'] = np.where(l1s > l1_threshold, 1, 0)
                    temp_df['impulsive'] = np.where(SNRs > SNR_threshold, 1, 0)

                    Flight.write_combined_scores_to_db(temp_df[['station_number', 'run_number', 'event_number', 'l1_max', 'amp_max', 'SNR_max', 'cw', 'impulsive']], filename[:-5])

            # save header information
            if len(header_df) == 0:
                header_df = temp_df
            else:
                header_df = pd.concat([header_df, temp_df], ignore_index=True, sort=False)

        # since we are processing whole file again in order to save the scores, we need to filter for desired time interval
        header_df = header_df[(header_df.trigger_time >= start_time.timestamp()) & (stop_time.timestamp() >= header_df.trigger_time)]
        header_df['index'] = range(0, len(header_df))
        if filetype == 'combined.root':
            header_df = header_df[['index', 'station_number', 'run_number', 'event_number', 'trigger_time', 'radiant_triggers', 'lt_triggers', 'l1_max', 'amp_max', 'SNR_max', 'cw', 'impulsive']] # change order to have index in front
        else:
            header_df = header_df[['index', 'station_number', 'run_number', 'event_number', 'trigger_time', 'radiant_triggers', 'lt_triggers']] # change order to have index in front

        return header_df



    #-------------------------------------------------------------------------------------------------------------------
    def write_combined_scores_to_db(df, filename='test', tablename='combined_scores'):
        path = f'./combined_scores/{filename}_scores.db'

        # Establish a connection to the SQLite database
        con = sqlite3.connect(path)
        
        # Write the DataFrame to the SQLite database
        df.to_sql(tablename, con, if_exists='append')
        
        # Close the database connection
        con.close()

    #------------------------------------------------------------------------------------------------------
    def plot_event_by_id(self, i):
        station_number = self.header_df.station_number.iloc[i]
        run_number = self.header_df.run_number.iloc[i]
        event_number = self.header_df.event_number.iloc[i]
        lt_trigger = self.header_df.lt_triggers.iloc[i]
        radiant_trigger = self.header_df.radiant_triggers.iloc[i]

        if lt_trigger == True:
            trigger_type = 'lt'
        elif radiant_trigger == True:
            trigger_type = 'radiant'
        else:
            trigger_type = 'Unknown'

        reader = readRNOGData()
        reader.begin([f'{Flight.path_to_combined_files}station{station_number}_run{run_number}_combined.root'], overwrite_sampling_rate=3200*units.MHz)

        evt = reader.get_event(run_nr=run_number, event_id=event_number)
        station = evt.get_station(station_number)
        

        fig, (ax0, ax1) = plt.subplots(2)
        fig.subplots_adjust(hspace=0.3)
        fig.suptitle(f'station: {station_number}, run: {run_number}, event_id: {event_number}, trigger: {trigger_type}, 24 channels')
        
        # setting labels
        ax0.plot([], [], label = 'trace')
        ax1.plot([], [], label = 'fourier transform')
        for i in range(24):
            channel = station.get_channel(i)
            trace = channel.get_trace()
            times = channel.get_times()
            spectrum = np.abs(channel.get_frequency_spectrum())
            freq = channel.get_frequencies()
            mask = (freq != 0.2) & (freq > 0.05) & (freq < 0.8)
            spectrum = spectrum[mask]
            freq = freq[mask]

            alpha = 0.5
            ax0.plot(times[:], trace[:], '-', alpha = alpha)
            ax1.plot(freq, spectrum, alpha = alpha)

        ax0.set_xlabel('time [ns]')
        ax0.set_ylabel('amplitude ~ [mV]')
        ax0.legend()
        ax1.set_xlabel('frequency [MHz]')
        ax1.set_ylabel('amplitude')
        ax1.legend()




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

        #print(pd.to_datetime(self.header_df.trigger_time.min(), unit = 's'), pd.to_datetime(self.header_df.trigger_time.max(), unit = 's'))
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

