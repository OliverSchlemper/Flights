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
from pandasql import sqldf
from rnog_data.runtable import RunTable
from datetime import datetime, timedelta

class Flights:

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
        return Flights.append_enu(stations, lon0 = stations.longitude.mean(), lat0 = stations.latitude.mean())

    #-------------------------------------------------------------------------------------------------------------------
    def process_db_files(start_time,
                        stop_time,
                        filedir='./data/',
                        destination='./flights/flights.db',
                        tablename=None,
                        R2=100,
                        append_min_max_time=True):
        """
        Processes SQLite database files containing aircraft information.
        Converts geodetic coordinates to local ENU coordinates, filters data based on radial distance,
        and writes the processed data back to the database.
        """
        if tablename is None:
            tablename = f'flights_{start_time}_{stop_time}'
        flights = flights_distinct = pd.DataFrame()
        
        # List SQLite database files in the specified directory
        files = [filename for filename in os.listdir(filedir) if '.db' in filename]
        
        # Iterate through each database file
        for filename in files:
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
        

        if len(flights) != 0:   
            # TBI
            # Append ENU coordinates to the DataFrame
            flights = Flights.append_enu(flights)
            
            flights['readtime_utc']= pd.to_datetime(flights.readtime, format='ISO8601').dt.tz_localize('Europe/London').dt.tz_convert('UTC')
        
            # Filter data for radial distance less than R2 (60 km)
            flights = flights[flights.r2 < R2**2]
        
            # Distinct flights
            flights_distinct = sqldf("SELECT distinct flightnumber, date, filename from flights")
        
            if append_min_max_time == True:
                flight_start_end_times = sqldf("SELECT min(readtime_utc) as mintime, max(readtime_utc) as maxtime, date, flightnumber from flights "
                                        "Group By date, flightnumber, filename")
                
                # Merge  start/end times to distinct flights
                flights_distinct = flights_distinct.merge(flight_start_end_times, on=['flightnumber', 'date'], how='left')

        
        # Write the updated DataFrame back to the database
        tablename_distinct = tablename + '_distinct'
        con = sqlite3.connect('./flights/flights.db')

        # Write the DataFrame to the SQLite database
        flights.to_sql(tablename, con, if_exists = 'replace')
        flights_distinct.to_sql(tablename_distinct, con, if_exists = 'replace')
        
        # Close the database connection
        con.close()
                    
        return 0

    #-------------------------------------------------------------------------------------------------------------------
    def get_rnog_data(filename,
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

        Returns:
            int: Returns 0 upon successful completion.

        Notes:
            - Requires wget utility to be installed.
            - Uses environment variables RNOG_USER_CHICAGO and RNOG_PASSWORD_CHICAGO 
            for authentication.
            - Adjusts directory structure while downloading to match the specified 
            path_data by cutting directories from the server URL.
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
    def get_flights(self,
                    tablename='flights',
                    db_file='flights'):
        tablename_distinct = tablename + '_distinct'
        
        con = sqlite3.connect(f'./flights/{db_file}.db')
        test = pd.read_sql_query(f"SELECT count(*) as length from {tablename} ", con)
        if test.length.iloc[0] == 0:
            table = table_distinct = pd.DataFrame()
        else:  
            table = pd.read_sql_query(f"Select * From {tablename} where date >= '{datetime.strftime(self.start_time, '%Y-%m-%d')}' and date < '{datetime.strftime(self.stop_time, '%Y-%m-%d')}' order by date", con)
            table_distinct = pd.read_sql_query(f"Select * From {tablename_distinct} where date >= '{datetime.strftime(self.start_time, '%Y-%m-%d')}' and date < '{datetime.strftime(self.stop_time, '%Y-%m-%d')}' order by date", con)
        
        con.close()
        return table, table_distinct
    #-------------------------------------------------------------------------------------------------------------------
    def get_flight_data(self):
        # get and unzip files
        current_time = self.start_time
        file_dates = sorted([s.split('-')[0] for s in os.listdir('./data/')])

        while current_time < self.stop_time:
            # increment time
            current_time += timedelta(days = 1)
            filename = str(datetime.strftime(current_time, '%Y.%m.%d')) + '-*.db.gz' # data for day x is in file for day x + 1, therefore use incremented current_time for filename

            if not filename.split('-')[0] in file_dates:
                Flights.get_rnog_data(filename=filename)
                #print(f"'./data/{filename.replace('.gz', '')}' already exists!")
                
        # Overwrite files with unzipped version
        os.system('cd data && gunzip --force *.gz > /dev/null 2>&1 && cd ..')
        #os.system('cd data && gunzip --force *.gz && cd ..')

        # process files to one big db file
        Flights.process_db_files(datetime.strftime(self.start_time.astimezone(Flights.london), Flights.fmt), datetime.strftime(self.stop_time.astimezone(Flights.london), Flights.fmt), tablename = 'flights')

    #-------------------------------------------------------------------------------------------------------------------
    def get_runtable(self):
        # getting runtable information and downloading header files
        self.runtable = self.rnogcopy() 
        #print(runtable)
        # check if files exits for flightnumber and time
        if len(self.runtable) == 0:
            sys.exit(f'No runs from "{str(self.start_time_plot)}" to "{str(self.stop_time_plot)}"')

        #prepare filtering on runtable
        self.runtable['run_string'] = 'run' + self.runtable.run.astype(str)
        self.runtable['station_string'] = 'station' + self.runtable.station.astype(str)

        # getting filenames for this flight
        self.filenames = []
        for i in range(len(self.runtable)):
            try:
                self.filenames.append([filename for filename in os.listdir('./header/') if re.search(self.runtable.station_string.iloc[i], filename) and re.search(self.runtable.run_string.iloc[i], filename)][0])
            except IndexError:
                print(f'No file with run {self.runtable.run.iloc[i]} and station {self.runtable.station.iloc[i]}')
        # read all header.root files in one DataFrame
        self.header_df = pd.DataFrame(columns = ['trigger_time', 'station_number', 'radiant_triggers'])
        for filename in self.filenames:
            self.file = uproot.open("header/" + filename)
            self.temp_df = pd.DataFrame(columns = ['trigger_time', 'station_number', 'radiant_triggers'])
            self.temp_df['trigger_time'] = np.array(self.file['header']['header/trigger_time'])
            self.temp_df['station_number'] = np.array(self.file['header']['header/station_number'])
            self.temp_df['radiant_triggers'] = np.array(self.file['header']['header/trigger_info/trigger_info.radiant_trigger'])
            self.temp_df['ext_triggers'] = np.array(self.file['header']['header/trigger_info/trigger_info.ext_trigger'])
            self.temp_df['force_triggers'] = np.array(self.file['header']['header/trigger_info/trigger_info.force_trigger']) 
            self.temp_df['lt_triggers'] = np.array(self.file['header']['header/trigger_info/trigger_info.lt_trigger'])
            
            if len(self.header_df) == 0:
                self.header_df = self.temp_df
            else:
                self.header_df = pd.concat([self.header_df, self.temp_df], ignore_index=True, sort=False)

        #print(header_df.trigger_time, datetime.strftime(start_time, fmt))
        self.header_df = self.header_df[(self.header_df.trigger_time >= self.start_time.timestamp()) & (self.stop_time.timestamp() >= self.header_df.trigger_time)]

    #-------------------------------------------------------------------------------------------------------------------
    def rnogcopy(self, filename='headers.root'):

        # get runtable
        self.rnog_table = RunTable()
        self.table = self.rnog_table.get_table(start_time=datetime.strftime(self.start_time_plot, '%Y-%m-%dT%H:%M:%S'), stop_time=datetime.strftime(self.stop_time_plot, '%Y-%m-%dT%H:%M:%S'))

        # check if files already exist in './header/'
        self.files_exist = True
        for i in range(len(self.table)):
            self.filename = 'station' + str(self.table.station.iloc[i]) + '_run' + str(self.table.run.iloc[i]) + '_headers.root'
            if not os.path.exists('./header/' + self.filename):
                self.files_exist = False
        
        if not self.files_exist:
            self.cmd = (f'''rnogcopy time "{str(datetime.strftime(self.start_time_plot, '%Y-%m-%dT%H:%M:%S'))}" "{str(datetime.strftime(self.stop_time_plot, '%Y-%m-%dT%H:%M:%S'))}" --filename=headers.root''')

            # "> /dev/null 2>&1" to suppress output 
            os.system("cd header && " + self.cmd + "> /dev/null 2>&1" + " && cd ..")
            #os.system("cd header && " + cmd + " && cd ..")
        
        #returning the information about the downloaded header files as a pandas dataframe
        return self.table

    #-------------------------------------------------------------------------------------------------------------------
    def plot_flight(self, i):
        self.index = i
        self.t = self.flights_distinct

        # variables
        self.flightnumber = self.t['flightnumber'].iloc[self.index]
        self.date = self.t.date.iloc[self.index]

        self.start_time_plot = self.t.mintime.iloc[self.index][:19] # [:19] to throw away potential microseconds
        self.stop_time_plot = self.t.maxtime.iloc[self.index][:19] # [:19] to throw away potential microseconds

        self.start_time_plot = self.utc.localize(datetime.strptime(self.start_time_plot, self.fmt))
        self.stop_time_plot = self.utc.localize(datetime.strptime(self.stop_time_plot, self.fmt))

        self.get_runtable()

        #------------------------------------------------------------------------------------------------------
        self.f = self.flights.query(f"readtime_utc >= '{datetime.strftime(self.start_time, self.fmt)}' & readtime_utc <= '{datetime.strftime(self.stop_time, self.fmt)}' & flightnumber == '{self.flightnumber}' ").copy()
        #f = flights_60.query(f"readtime >= '{start_timestamp}' & readtime <= '{stop_timestamp}'").copy()
        self.times = pd.to_datetime(self.f.readtime_utc, format='ISO8601').astype('int64') / 10**9
        self.r = np.sqrt(self.f.r2)

        print('''"''' ,self.start_time_plot, '''"''', '''"''' ,self.stop_time_plot, '''"''' + f' duration: {self.stop_time_plot - self.start_time_plot} [s]')
        print(self.flightnumber)


        # plot data
        plt.rcParams.update({'font.size': 5})
        self.fig, self.ax = plt.subplots(3,3, figsize=(8.27, 11.69), dpi=100)
        self.fig.subplots_adjust(hspace=0.3, wspace=0.4)
        self.fig.suptitle(self.flightnumber + ', ' + self.date + ''', "''' + str(self.start_time_plot) + '''", "''' + str(self.stop_time_plot) + '''"''')

        #------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------
        # ax[0]
        self.ax[0, 0].set_xlabel('longitude [deg]')
        self.ax[0, 0].set_ylabel('latitude [deg]')
        self.ax[0, 0].set_title('trajectory')
        #ax[0, 0].set_xlim(-39, -38)
        #ax[0, 0].set_ylim(72.5, 72.7)

        #------------------------------------------------------------------------------------------------------
        # set ticks for time colorbar
        self.ticks = np.linspace(self.start_time_plot.timestamp(), self.stop_time_plot.timestamp(), 8)
        self.tick_times = pd.to_datetime(self.ticks, unit = 's').strftime('%H:%M:%S')


        #------------------------------------------------------------------------------------------------------
        # stations
        for i in range(len(self.stations)):
            self.ax[0, 0].scatter(self.stations.longitude[i], self.stations.latitude[i], marker = 'x', label = self.stations['Station Name'][i])

        self.sc = self.ax[0, 0].scatter(self.f.longitude, self.f.latitude, marker = '.', c = self.times, cmap = 'viridis')
        self.cbar = self.fig.colorbar(self.sc, ax=self.ax[0, 0])
        self.cbar.set_ticks(self.ticks)
        self.cbar.set_ticklabels(self.tick_times)

        self.ax[0, 0].legend()

        #------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------
        # ax[1]
        self.n_bins = np.arange(self.start_time_plot.timestamp(), self.stop_time_plot.timestamp(), 10)

        self.ax[0, 1].hist(self.header_df[self.header_df.lt_triggers == True].trigger_time, bins = self.n_bins, color = 'C0', label = 'lt triggers', histtype = 'step')
        self.ax[0, 1].hist(self.header_df[self.header_df.radiant_triggers == True].trigger_time, bins = self.n_bins, color = 'C1', label = 'radiant triggers', histtype = 'step')


        self.ax_01_twin = self.ax[0, 1].twinx()
        self.ax_01_twin.plot(self.times, self.r, '.', markersize = 3, label = 'd [km]', color = 'C4')
        #ax_01_twin.plot(times, f.altitude/1000, 'x', color = 'C5')
        self.ax_01_twin.plot(self.times, self.f.z, '.', markersize = 3, label = 'altitude [km]', color = 'C6')
        #ax_01_twin.plot(times, np.sqrt(f.x**2 + f.y**2 + f.z**2), 'x', color = 'C7')

        self.ax[0, 1].set_title('Sum all stations')
        self.ax[0, 1].set_xticks(self.ticks)
        self.ax[0, 1].set_xticklabels(self.tick_times, rotation=90)
        self.ax[0, 1].set_xlim(min(self.ticks), max(self.ticks))
        self.ax[0, 1].legend()
        self.ax_01_twin.legend(loc = 1)

        #------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------
        self.counter = 0
        for self.i in range(3):
            for self.j in range(3):
                
                if self.i > 0 or self.j > 1:
                    self.nr = self.stations['Station Nr.'].iloc[self.counter - 1]
                    self.twinx = self.ax[self.i, self.j].twinx()
                    self.axes = self.ax[self.i, self.j]
                    
                    self.axes.set_title(f'Station {self.nr}')
                    self.axes.set_xticks(self.ticks)
                    self.axes.set_xticklabels(self.tick_times, rotation=90)
                    self.axes.set_xlim(min(self.ticks), max(self.ticks))
                    if self.j > 1:
                        self.twinx.set_ylabel('d [km]')
                    if self.i > 0:
                        self.axes.sharey(self.ax[0, 2])
                    if self.i > 1:
                        self.axes.set_xlabel('time (hh:mm:ss)')
                    if self.j == 0:
                        self.axes.set_ylabel('# triggers / 10s')
                    
                    self.temp_f = Flights.append_enu(self.f, self.stations[self.stations['Station Nr.'] == self.nr]['longitude'].to_numpy()[0], self.stations[self.stations['Station Nr.'] == self.nr]['latitude'].to_numpy()[0])
                    self.twinx.plot(self.times, np.sqrt(self.temp_f.r2), '.', markersize = 3, color = 'C4')
                    self.twinx.plot(self.times, self.f.z, '.', markersize = 3, label = 'altitude [km]', color = 'C6')
                    self.axes.hist(self.header_df[(self.header_df['station_number'] == self.nr) & self.header_df['radiant_triggers'] == True].trigger_time, bins = self.n_bins, histtype = 'step', color = 'C1')
                    self.axes.hist(self.header_df[(self.header_df['station_number'] == self.nr) & self.header_df['lt_triggers'] == True].trigger_time, bins = self.n_bins, histtype = 'step', color = 'C0')
                    #ax[i, j].legend()
                self.counter = self.counter + 1


    def show_flights(self):
        print(self.flights_distinct[['flightnumber', 'date', 'filename']])


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
    #-------------------------------------------------------------------------------------------------------------------
    

    # time zones
    london = pytz.timezone('Europe/London')
    utc = pytz.timezone('UTC')

    # time format
    fmt = '%Y-%m-%d %H:%M:%S'

    def __init__(self, start_time, stop_time):

        self.start_time = self.utc.localize(datetime.strptime(start_time, self.fmt))
        self.stop_time = self.utc.localize(datetime.strptime(stop_time, self.fmt))

        # initialize stations dataframe
        self.stations = Flights.init_stations()
        self.get_flight_data()
        self.flights, self.flights_distinct = self.get_flights()

    

    