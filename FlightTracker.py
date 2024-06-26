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

from IPython.display import clear_output

from Flight import Flight

class FlightTracker:

    # functions
    #-------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------
    def append_enu(dataframe, lon0=-38.45, lat0=72.58, z0=0):
        """
        Processes a pandas DataFrame by converting geodetic coordinates to local East-North-Up (ENU) coordinates,
        normalizing coordinates, calculating radial distance squared, and returning the updated DataFrame.

        Returns:
            DataFrame: The updated pandas DataFrame with processed coordinates and radial distance squared.
        """
        # Convert geodetic coordinates to local ENU coordinates
        dataframe['x'], dataframe['y'], dataframe['z'] = pm.geodetic2enu(dataframe.longitude, dataframe.latitude, dataframe.altitude*0.3048, lon0, lat0, z0) # altitude from foot to meters
        
        # convert m to km
        dataframe['x'], dataframe['y'], dataframe['z'] = dataframe['x']/1000, dataframe['y']/1000, dataframe['z']/1000
        
        # Calculate radial distance squared
        dataframe['r2'] = dataframe.x**2 + dataframe.y**2 + dataframe.z**2
        
        return dataframe


    #-------------------------------------------------------------------------------------------------------------------
    def append_l1():
        pass
    #-------------------------------------------------------------------------------------------------------------------
    def init_stations():
        s = [['Big House', -38.45, 72.58, 0]
             ,['Station 11', -38.502299 , 72.589227, 11]
             ,['Station 12', -38.496227 , 72.600087, 12]
             ,['Station 13', -38.490147 , 72.610947, 13]
             ,['Station 21', -38.466030 , 72.587406, 21]
             ,['Station 22', -38.459936 , 72.598265, 22]
             ,['Station 23', -38.453833 , 72.609124, 23]
             ,['Station 24', -38.447723 , 72.619983, 24]
            ]

        #-------------------------------------------------------------------------------------------------------------------
        stations = pd.DataFrame(s, columns=['Station Name', 'longitude', 'latitude', 'Station Nr.'])
        stations['altitude'] = 0
        #-------------------------------------------------------------------------------------------------------------------
        #lon0, lat0 = stations[stations['Station Name'] == 'Big House'][['longitude', 'latitude']].to_numpy()[0]
        return FlightTracker.append_enu(stations, lon0 = stations.longitude.mean(), lat0 = stations.latitude.mean())

    #-------------------------------------------------------------------------------------------------------------------
    def process_db_files(start_time,
                        stop_time,
                        filedir='./data/',
                        destination='./flights/flights.db',
                        tablename='flights',
                        R2=150,
                        append_min_max_time=True):
        """
        Processes SQLite database files containing aircraft information.
        Converts geodetic coordinates to local ENU coordinates, filters data based on radial distance,
        and writes the processed data back to the database.
        """

        flights = flights_distinct = df = pd.DataFrame()

        # List SQLite database files in the specified directory
        files = [filename for filename in os.listdir(filedir) if '.db' in filename] 
        # Iterate through each database file
        for filename in files:
            print(f'start {filename}')
            # Establish connection to the database file
            con = sqlite3.connect(filedir + filename)

            #check if table 'aircraft' exists
            table_name_aircraft = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' and name = 'aircraft'", con)['name']

            if len(table_name_aircraft) == 0:
                print(f"No table 'aircraft' in file: {filename}")
            else:
                # Read data from'aircraft' table into a pandas DataFrame
                df = pd.read_sql_query(f"SELECT *, date(readtime) as date from aircraft Where readtime >= '{start_time}' And readtime < '{stop_time}'", con)
                df['filename'] = filename
            # Close the database connection
            con.close()

            if(len(df)) != 0:
                flights = pd.concat([flights, df], ignore_index=True)
            print(f'end {filename}')
            clear_output(wait=True)
        

        if len(flights) != 0:   
            # TBI
            # Append ENU coordinates to the DataFrame
            flights = FlightTracker.append_enu(flights)
            
            flights['readtime_utc']= pd.to_datetime(flights.readtime, format='ISO8601').dt.tz_localize('Europe/London').dt.tz_convert('UTC')
        
            # Filter data for radial distance less than R2 (60 km)
            flights = flights[flights.r2 < R2**2]
        
            # Distinct flights
            flights_distinct = sqldf("SELECT distinct flightnumber, date, filename from flights")
        
            if append_min_max_time == True:
                flight_start_end_times = sqldf( ''' SELECT 
                                                        min(readtime_utc) as mintime, 
                                                        max(readtime_utc) as maxtime, 
                                                        round(sqrt(min(r2)), 1) as min_r, 
                                                        round(min(z), 1) as min_z, date, 
                                                        flightnumber, 
                                                        min(x) as min_x, 
                                                        max(x) as max_x,
                                                        min(y) as min_y,
                                                        max(y) as max_y
                                                    from flights 
                                                    Group By date, flightnumber, filename''')
                
                # Merge  start/end times to distinct flights
                flights_distinct = flights_distinct.merge(flight_start_end_times, on=['flightnumber', 'date'], how='left')
                flights_distinct['theta'] = np.round(np.rad2deg(np.arctan2(flights_distinct.max_y - flights_distinct.min_y, flights_distinct.max_x - flights_distinct.min_x)))
                flights_distinct.drop(columns = ['min_x', 'max_x', 'min_y', 'max_y'], inplace = True)
        
        # Write the updated DataFrame back to the database
        tablename_distinct = tablename + '_distinct'
        con = sqlite3.connect(destination)

        # Write the DataFrame to the SQLite database
        flights.to_sql(tablename, con, if_exists = 'replace')
        flights_distinct.to_sql(tablename_distinct, con, if_exists = 'replace')
        
        # Close the database connection
        con.close()
                    
        return 0

    #-------------------------------------------------------------------------------------------------------------------
    def download_flight_tracker_db_files(filename,
                                        path_data='./data/',
                                        server_url='https://rno-g.uchicago.edu/data/flight-tracker/'):
        """
        Download RNO-G data file(s) from the specified server URL.

        Parameters:
            filename (str): Name of the file(s) to be downloaded. 
                            Supports wildcards (*).
            path_data (str): Destination directory where the file(s) 
                            will be saved. Defaults to current directory.
            server_url (str): URL of the RNO-G server from which to download 
                            the data. Defaults to 'https://rno-g.uchicago.edu/data/flight-tracker/'.

        Notes:
            - Requires wget utility to be installed.
            - Uses environment variables RNOG_USER_CHICAGO and RNOG_PASSWORD_CHICAGO 
            for authentication.
        """

        # Determine the number of directories to cut from the server URL
        dir_cut_length = len(server_url.split('/')[3:-1])

        # Construct wget command to download the file(s)
        cmd = (f"wget -q -r -np -A '{filename}' -nH --cut-dirs {dir_cut_length} -P {path_data} "
            f"--user $RNOG_USER_CHICAGO --password $RNOG_PASSWORD_CHICAGO {server_url}")

        # Execute the wget command
        os.system(cmd)

        return 0

    #-------------------------------------------------------------------------------------------------------------------
    def get_flights_and_flights_distinct(start_time,
                    stop_time,
                    tablename='flights',
                    destination='flights/flights.dnb'):
        tablename_distinct = tablename + '_distinct'
        
        con = sqlite3.connect(f'{destination}')
        test = pd.read_sql_query(f"SELECT count(*) as length from {tablename} ", con)
        if test.length.iloc[0] == 0:
            table = table_distinct = pd.DataFrame() # if there is nothing in db file return empty dataframe(s)
        else:  
            table = pd.read_sql_query(f"Select * From {tablename} where date >= '{datetime.strftime(start_time, '%Y-%m-%d')}' and date < '{datetime.strftime(stop_time, '%Y-%m-%d')}' order by date", con)
            table_distinct = pd.read_sql_query(f"Select * From {tablename_distinct} where date >= '{datetime.strftime(start_time, '%Y-%m-%d')}' and date < '{datetime.strftime(stop_time, '%Y-%m-%d')}' order by date", con) 
        con.close()
        return table, table_distinct

    #-------------------------------------------------------------------------------------------------------------------
    def download_and_process_db_files(start_time, stop_time, destination):
        # get and unzip files
        current_time = start_time
        file_dates = sorted([s.split('-')[0] for s in os.listdir('./data/')])

        filenames_flight_tracker_db = []

        while current_time < stop_time:
            # increment time
            current_time += timedelta(days = 1)
            filename = str(datetime.strftime(current_time, '%Y.%m.%d')) + '-*.db.gz' # data for day x is in file for day x + 1, therefore use incremented current_time for filename
            filenames_flight_tracker_db.append(filename)

            # only get rnog data if files don't already exist in './data/'
            if not filename.split('-')[0] in file_dates:
                FlightTracker.download_flight_tracker_db_files(filename=filename)
                #print(f"'./data/{filename.replace('.gz', '')}' already exists!")
                
        # Overwrite files with unzipped version
        os.system('cd data && gunzip --force *.gz > /dev/null 2>&1 && cd ..')

        # process files to one big db file
        FlightTracker.process_db_files(datetime.strftime(start_time.astimezone(FlightTracker.london), FlightTracker.fmt), datetime.strftime(stop_time.astimezone(FlightTracker.london), FlightTracker.fmt), destination = destination, tablename = 'flights')

    #-------------------------------------------------------------------------------------------------------------------
    def get_flight_by_index(self, i, filetype='combined.root'):
        return Flight(self, i, filetype=filetype)
    
    #-------------------------------------------------------------------------------------------------------------------
    def set_flight_index(self, i):
        index = i
        self.t = self.flights_distinct

        # variables
        self.flightnumber = self.t['flightnumber'].iloc[index]
        self.date = self.t.date.iloc[index]

        self.start_time_plot = self.t.mintime.iloc[index][:19] # [:19] to throw away potential microseconds
        self.stop_time_plot = self.t.maxtime.iloc[index][:19] # [:19] to throw away potential microseconds

        self.start_time_plot = FlightTracker.utc.localize(datetime.strptime(self.start_time_plot, FlightTracker.fmt))
        self.stop_time_plot = FlightTracker.utc.localize(datetime.strptime(self.stop_time_plot, FlightTracker.fmt))

        self.header_df = FlightTracker.get_df_from_root_file(self.start_time_plot, self.stop_time_plot)
        
        self.f = self.flights.query(f"readtime_utc >= '{datetime.strftime(self.start_time_plot, FlightTracker.fmt)}' & readtime_utc <= '{datetime.strftime(self.stop_time_plot, FlightTracker.fmt)}' & flightnumber == '{self.flightnumber}' ").copy()
        #f = flights_60.query(f"readtime >= '{start_timestamp}' & readtime <= '{stop_timestamp}'").copy()
        self.times = pd.to_datetime(self.f.readtime_utc, format='ISO8601').astype('int64') / 10**9
        self.r = np.sqrt(self.f.r2)


        print('''"'''  + self.start_time_plot.strftime("%Y-%m-%dT%H:%M:%S") + '''"''', '''"''' + self.stop_time_plot.strftime("%Y-%m-%dT%H:%M:%S") +  '''"''' + f' duration: {self.stop_time_plot - self.start_time_plot} [s]')
        print(self.flightnumber)

    #-------------------------------------------------------------------------------------------------------------------
    def get_df_from_root_file(start_time, stop_time, file = 'headers.root'):
        # getting runtable information and downloading header files
        runtable = FlightTracker.rnogcopy(start_time, stop_time, file) 
        # check if files exits for flightnumber and time
        if len(runtable) == 0:
            sys.exit(f'No runs from "{str(start_time)}" to "{str(stop_time)}"')

        #prepare filtering on runtable
        runtable['run_string'] = 'run' + runtable.run.astype(str)
        runtable['station_string'] = 'station' + runtable.station.astype(str)

        # getting filenames for this flight
        filenames = []
        for i in range(len(runtable)):
            try:
                filenames.append([filename for filename in os.listdir('./header/') if re.search(runtable.station_string.iloc[i], filename) and re.search(runtable.run_string.iloc[i], filename)][0])
            except IndexError:
                print(f'No file with run {runtable.run.iloc[i]} and station {runtable.station.iloc[i]}')
        # read all headers.root files in one DataFrame
        header_df = pd.DataFrame(columns = ['trigger_time', 'station_number', 'radiant_triggers'])
        for filename in filenames:
            file = uproot.open("header/" + filename)
            temp_df = pd.DataFrame(columns = ['trigger_time', 'station_number', 'radiant_triggers'])
            temp_df['trigger_time'] = np.array(file['header']['header/trigger_time'])
            temp_df['station_number'] = np.array(file['header']['header/station_number'])
            temp_df['radiant_triggers'] = np.array(file['header']['header/trigger_info/trigger_info.radiant_trigger'])
            #temp_df['ext_triggers'] = np.array(file['header']['header/trigger_info/trigger_info.ext_trigger'])
            #temp_df['force_triggers'] = np.array(file['header']['header/trigger_info/trigger_info.force_trigger']) 
            temp_df['lt_triggers'] = np.array(file['header']['header/trigger_info/trigger_info.lt_trigger'])
            
            if len(header_df) == 0:
                header_df = temp_df
            else:
                header_df = pd.concat([header_df, temp_df], ignore_index=True, sort=False)

        #print(header_df.trigger_time, datetime.strftime(start_time, fmt))
        header_df = header_df[(header_df.trigger_time >= start_time.timestamp()) & (stop_time.timestamp() >= header_df.trigger_time)]
        return header_df

    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    def rnogcopy(start_time, stop_time, file='headers.root'):

        #check for filetype
        if file == 'headers.root':
            path = 'header'
        elif file == 'combined.root':
            path = 'combined'
        else:
            print(f'Unknown filename: {file}')

        # get runtable
        rnog_table = RunTable()
        table = rnog_table.get_table(start_time=datetime.strftime(start_time, '%Y-%m-%dT%H:%M:%S'), stop_time=datetime.strftime(stop_time, '%Y-%m-%dT%H:%M:%S'))

        # check if files already exist in './header/'
        files_exist = True
        for i in range(len(table)):
            filename = 'station' + str(table.station.iloc[i]) + '_run' + str(table.run.iloc[i]) + f'_{file}'
            if not os.path.exists(f'./{path}/{filename}'):
                files_exist = False
        
        if not files_exist:
            cmd = (f'''rnogcopy time "{str(datetime.strftime(start_time, '%Y-%m-%dT%H:%M:%S'))}" "{str(datetime.strftime(stop_time, '%Y-%m-%dT%H:%M:%S'))}" --filename={file}''')

            # "> /dev/null 2>&1" to suppress output 
            os.system(f"cd {path} && " + cmd + "> /dev/null 2>&1" + " && cd ..")

        #returning the information about the downloaded header files as a pandas dataframe
        return table
    #-------------------------------------------------------------------------------------------------------------------
    def part_lin(x, times, r):
        return_linspace = []
        for element in x:
            index = np.where(times <= element)[0][-1]
            if index >= (len(times) - 1):
                index = index - 1
                print(f'Index {index} out of range {len(times) - 1}')
            y2 = r.iloc[index+1]
            y1 = r.iloc[index]
            x2 = times.iloc[index+1]
            x1 = times.iloc[index]
            m = (y2 - y1) / (x2 - x1)
            t = y2 - m * x2
            return_linspace.append(m*element+t)
        len_ret = len(return_linspace)
        len_x = len(x)
        while(len_ret < len_x):
            return_linspace.append(0)

        return return_linspace

    #-------------------------------------------------------------------------------------------------------------------
    def plot_flight(self, i):
        
        self.set_flight_index(i)

        # plot data
        plt.rcParams.update({'font.size': 5})
        self.fig, self.ax = plt.subplots(3,3, figsize=(8.27, 11.69), dpi=100)
        self.fig.subplots_adjust(hspace=0.3, wspace=0.4)
        self.fig.suptitle(self.flightnumber + ', ' + self.date + ''', "''' + str(self.start_time_plot) + '''", "''' + str(self.stop_time_plot) + '''"''')

        #------------------------------------------------------------------------------------------------------
        #-----------------------------------------------------------------------filedir-------------------------------
        # ax[0]
        self.ax[0, 0].set_xlabel('longitude [deg]')
        self.ax[0, 0].set_ylabel('latitude [deg]')
        self.ax[0, 0].set_title('trajectory')
        #ax[0, 0].set_xlim(-39, -38)
        #ax[0, 0].set_ylim(72.5, 72.7)

        #------------------------------------------------------------------------------------------------------
        # set ticks for time colorbar
        ticks = np.linspace(self.start_time_plot.timestamp(), self.stop_time_plot.timestamp(), 8)
        tick_times = pd.to_datetime(ticks, unit = 's').strftime('%H:%M:%S')


        #------------------------------------------------------------------------------------------------------
        # stations
        for i in range(len(self.stations)):
            self.ax[0, 0].scatter(self.stations.longitude[i], self.stations.latitude[i], marker = 'x', label = self.stations['Station Name'][i], s = 1)

        sc = self.ax[0, 0].scatter(self.f.longitude, self.f.latitude, marker = '.', c = self.times, cmap = 'viridis', s = 1)
        cbar = self.fig.colorbar(sc, ax=self.ax[0, 0])
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_times)

        self.ax[0, 0].legend()

        #------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------
        # ax[1]
        self.n_bins = np.arange(self.start_time_plot.timestamp(), self.stop_time_plot.timestamp(), 10)
        n_bins = self.n_bins

        self.ax[0, 1].hist(self.header_df[self.header_df.lt_triggers == True].trigger_time, bins = n_bins, color = 'C0',  label = 'lt triggers', histtype = 'step', linewidth = 1, alpha = 0.5)
        self.ax[0, 1].hist(self.header_df[self.header_df.radiant_triggers == True].trigger_time, bins = n_bins, color = 'C1',  label = 'radiant triggers', histtype = 'step', linewidth = 1, alpha = 0.5)


        self.ax_01_twin = self.ax[0, 1].twinx()
        self.ax_01_twin.plot(self.times, self.r, '.', markersize = 1, label = 'd [km]', color = 'C4')

        x = np.linspace(self.start_time_plot.timestamp(), self.stop_time_plot.timestamp(), 100)
        #self.ax_01_twin.plot(x[1:-1], FlightTracker.part_lin(x[1:-1], times, r), '-')
        #ax_01_twin.plot(times, f.altitude/1000, 'x', color = 'C5')
        self.ax_01_twin.plot(self.times, self.f.z, '.', markersize = 1, label = 'altitude [km]', color = 'C6')
        #ax_01_twin.plot(times, np.sqrt(f.x**2 + f.y**2 + f.z**2), 'x', color = 'C7')

        self.ax[0, 1].set_title('Sum all stations')
        self.ax[0, 1].set_xticks(ticks)
        self.ax[0, 1].set_xticklabels(tick_times, rotation=90)
        self.ax[0, 1].set_xlim(min(ticks), max(ticks))
        self.ax[0, 1].legend()
        self.ax_01_twin.legend(loc = 1)



        #------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------
        counter = 0
        for i in range(3):
            for j in range(3):
                
                if i > 0 or j > 1:
                    nr = self.stations['Station Nr.'].iloc[counter - 1]
                    twinx = self.ax[i, j].twinx()
                    axes = self.ax[i, j]
                    
                    axes.set_title(f'Station {nr}')
                    axes.set_xticks(ticks)
                    axes.set_xticklabels(tick_times, rotation=90)
                    axes.set_xlim(min(ticks), max(ticks))
                    if j > 1:
                        twinx.set_ylabel('d [km]')
                    if i> 0:
                        axes.sharey(self.ax[0, 2])
                    if i> 1:
                        axes.set_xlabel('time (hh:mm:ss)')
                    if j == 0:
                        axes.set_ylabel('# triggers / 10s')
                    
                    temp_f = FlightTracker.append_enu(self.f, self.stations[self.stations['Station Nr.'] == nr]['longitude'].to_numpy()[0], self.stations[self.stations['Station Nr.'] == nr]['latitude'].to_numpy()[0])
                    twinx.plot(self.times, np.sqrt(temp_f.r2), '.', markersize = 1, color = 'C4')
                    twinx.plot(self.times, self.f.z, '.', markersize = 1, label = 'altitude [km]', color = 'C6')
                    axes.hist(self.header_df[(self.header_df['station_number'] == nr) & self.header_df['radiant_triggers'] == True].trigger_time, bins = n_bins,  histtype = 'step', color = 'C1', alpha = 0.5)
                    axes.hist(self.header_df[(self.header_df['station_number'] == nr) & self.header_df['lt_triggers'] == True].trigger_time, bins = n_bins,  histtype = 'step', color = 'C0', alpha = 0.5)
                    #ax[i, j].legend()
                counter += 1


    #-------------------------------------------------------------------------------------------------------------------
    def show_flights(self):
        self.flights_distinct[['flightnumber', 'date', 'filename', 'min_r', 'mintime', 'maxtime']].head(20)

        #print(self.flights_distinct[['flightnumber', 'date', 'filename', 'min_r', 'mintime', 'maxtime']])

    #-------------------------------------------------------------------------------------------------------------------


    #-------------------------------------------------------------------------------------------------------------------
    def plot_trigger_rate_over_d(self):

        f = self.flights.query(f"readtime_utc >= '{datetime.strftime(self.start_time, FlightTracker.fmt)}' & readtime_utc <= '{datetime.strftime(self.stop_time, FlightTracker.fmt)}' & flightnumber == '{self.flightnumber}' ").copy()
        #f = flights_60.query(f"readtime >= '{start_timestamp}' & readtime <= '{stop_timestamp}'").copy()
        times = pd.to_datetime(f.readtime_utc, format='ISO8601').astype('int64') / 10**9
        r = np.sqrt(f.r2)

        self.fig2, self.ax2 = plt.subplots(1, 2)
        self.fig2.suptitle('trigger rate over d [km]')

        x = np.linspace(times.min(), times.max(), 100)[1:-1]

        # set ticks
        ticks = np.linspace(times.min(), times.max(), 8)
        tick_times = pd.to_datetime(ticks, unit = 's').strftime('%H:%M:%S')

        self.ax2_0_twin = self.ax2[0].twinx()

        self.ax2[0].hist(self.header_df[self.header_df.lt_triggers == True].trigger_time, bins = self.n_bins, color = 'C0',  label = 'lt triggers', histtype = 'step', alpha = 0.5)
        self.ax2[0].hist(self.header_df[self.header_df.radiant_triggers == True].trigger_time, bins = self.n_bins, color = 'C1',  label = 'radiant triggers', histtype = 'step', alpha = 0.5)


        self.ax2[0].set_xticks(ticks)
        self.ax2[0].set_xticklabels(tick_times, rotation=90)

        self.ax2[0].set_title('d [km] over time')
        self.ax2_0_twin.set_xlabel('time')
        self.ax2_0_twin.set_ylabel('d [km]')
        self.ax2_0_twin.plot(times, r, '.', markersize = 1, label = 'd [km]', color = 'C4')
        self.ax2_0_twin.plot(times, f.z, '.', markersize = 1, label = 'altitude [km]', color = 'C6')
        self.ax2_0_twin.plot(x, FlightTracker.part_lin(x, times, r), '--', label = 'lin fit', color = 'C5', markersize = 1)
        self.ax2_0_twin.legend()


        self.ax2[1].set_xlabel('d [km]')
        self.ax2[1].set_ylabel('# triggers / 5 km')
        self.ax2[1].hist(FlightTracker.part_lin(self.header_df[(times.min() <= self.header_df.trigger_time) & (self.header_df.trigger_time <= times.max())].trigger_time, times, r),  histtype = 'step', linewidth= 1, bins = 30, alpha = 0.5)
    #-------------------------------------------------------------------------------------------------------------------
    def storage():
           # Check if flight data is available for time and output table
        if len(flights) == 0:
            print(f'No flight data between {str(start_time)} - {str(stop_time)}')
        else:
            if 'index' in flights_distinct.columns:
                flights_distinct.drop(columns = ['index'], inplace = True)
            IPython.display.clear_output
            print(flights_distinct.head(50))


    #-------------------------------------------------------------------------------------------------------------------
    def create_dirs():
        # data for flight_tracker data
        if not os.path.exists('./data/'):
            os.system('mkdir data')
        # flights for db files containing processed flight_tracker data
        if not os.path.exists('./flights/'):
            os.system('mkdir flights')
        # header files
        if not os.path.exists('./header/'):
            os.system('mkdir header')
        # combined files
        if not os.path.exists('./combined/'):
            os.system('mkdir combined')
        # combined scores (l1, max_amplitude, SNR)
        if not os.path.exists('./combined_scores/'):
            os.system('mkdir combined_scores')
    #-------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------
    

    # time zones
    london = pytz.timezone('Europe/London')
    utc = pytz.timezone('UTC')

    # time format
    fmt = '%Y-%m-%d %H:%M:%S'

    def __init__(self, start_time, stop_time=None, destination='./flights/flights.db', already_calculated=False):
        #make dirs
        FlightTracker.create_dirs()

        self.start_time = FlightTracker.utc.localize(datetime.strptime(start_time, FlightTracker.fmt))
        if stop_time == None:
            self.stop_time = self.start_time + timedelta(days=1)
        else:
            self.stop_time = FlightTracker.utc.localize(datetime.strptime(stop_time, FlightTracker.fmt))

        # initialize stations dataframe
        self.stations = FlightTracker.init_stations()
        if already_calculated == False:
            FlightTracker.download_and_process_db_files(self.start_time, self.stop_time, destination) # downloads flight tracker data and saves a db file with flights / flights_distinct
        self.flights, self.flights_distinct = FlightTracker.get_flights_and_flights_distinct(self.start_time, self.stop_time, destination=destination) # load flights / flights_distinct from a db file