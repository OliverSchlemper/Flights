import argparse
import pytz
import os
import sqlite3
import re
import uproot
import IPython
import sys
import json
import numpy as np
import pandas as pd
import pymap3d as pm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import hilbert
from pandasql import sqldf
from rnog_data.runtable import RunTable
from datetime import datetime, timedelta
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.utilities import units

from IPython.display import clear_output

from Flight import Flight

HANDCARRY_DATA = 'combined_handcarry'
SCORES = 'scores_from_cluster'

#------------------------------------------------------------------------------------------------------
def calculate_avg_RMS(event, station_number):
    station = event.get_station(station_number)
    RMSs = np.zeros(24) # save avg for each channel here
    for i in range(24):
        channel = station.get_channel(i)
        trace = channel.get_trace()
        RMSs[i] = np.sqrt(np.mean(trace**2))
    return RMSs

#------------------------------------------------------------------------------------------------------
def impulsivity(trace, start_index=0, debug=False):
    ''' impulsivity metric based on the excess in Hilbert-envelope close to the highest amplitude pulse '''
    abs_hilbert = abs(hilbert(trace))
    hilbert_maximum_index = np.argmax(abs_hilbert)
    reordered_index = np.argsort(abs(np.arange(len(trace))-hilbert_maximum_index))
    reordered_hilbert = abs_hilbert[reordered_index]
    impulsivity_curve = np.cumsum(reordered_hilbert[start_index:])/np.sum(reordered_hilbert[start_index:])
    impulsivity = 2*np.mean(impulsivity_curve)-1    
    if debug:
        plt.plot(np.arange(len(trace[start_index:])), impulsivity_curve, label=f"impulsivity: {impulsivity:.5f}")
        plt.fill_between(np.arange(len(trace[start_index:])), np.cumsum(np.ones_like(trace[start_index:])/len(trace[start_index:])), impulsivity_curve, alpha=0.4)
        plt.xlabel("sample")
        plt.ylabel("normalized pulse-ordered CDF")
        plt.legend()
        plt.xlim(0, len(trace[start_index:]))
        plt.ylim(0,1)
        plt.plot([0,len(trace[start_index:])],[0,1], ":", color="black")
    return max(impulsivity,0)

#------------------------------------------------------------------------------------------------------ 
def simple_l1(frequencies):
    return np.max(frequencies**2)/np.sum(frequencies**2)

#------------------------------------------------------------------------------------------------------
def calc_l1_amp_SNR(event, station_number, avg_RMS):
    #print(station_number)
    l1s = np.zeros(24) 
    amps = np.zeros(24) 
    SNRs = np.zeros(24) 
    RMSs = np.zeros(24)
    impulsivities = np.zeros(24)
    max_freqs = np.zeros(24)
    max_spectrums = np.zeros(24)

    station = event.get_station(station_number)
    for i in range(24):
        channel = station.get_channel(i)
        trace = np.array(channel.get_trace(), dtype = float)
        times = channel.get_times()
        #times_mask = (times < 0)

        freq = channel.get_frequencies()
        mask = (0.05 < freq) & (freq < 0.8) & (freq != 0.2)
        freq = freq[mask]
        spectrum = np.abs(channel.get_frequency_spectrum())[mask]
        # get the freq with max amplitude
        max_spectrum = max(spectrum)
        mask_max_freq = [spectrum == max_spectrum][0]

        max_freqs[i] = freq[mask_max_freq][0]
        max_spectrums[i] = max_spectrum

        #calculate
        l1s[i] = simple_l1(spectrum)
        amps[i] = np.max(np.abs(trace))
        impulsivities[i] = impulsivity(trace)
        #avg = np.average(trace)
        #RMS = np.sqrt(np.mean(trace[times_mask]**2))
        SNRs[i] = amps[i] / avg_RMS[i]

    return l1s, amps, SNRs, RMSs, impulsivities, max_freqs, max_spectrums

#------------------------------------------------------------------------------------------------------
def write_combined_scores_to_db(df, filename, tablename):
    path = f'./{SCORES}/{filename}_scores.db'

    # Establish a connection to the SQLite database
    con = sqlite3.connect(path)
    
    # Write the DataFrame to the SQLite database
    df.to_sql(tablename, con, if_exists='replace')
    
    # Close the database connection
    con.close()

#------------------------------------------------------------------------------------------------------
def process_scores( station, run):
    filepath = f'station{station}/run{run}'
    filename = f'station{station}_run{run}'

    #header
    #------------------------------------------------------------------------------------------------------
    path = 'header' # didn't want to change this in the following rows so just kept path a variable
    file = uproot.open(f'./{HANDCARRY_DATA}/{filepath}/headers.root')
    temp_df = pd.DataFrame()
    temp_df['station_number'] = np.array(file[path]['header/station_number'])
    temp_df['run_number'] = np.array(file[path]['header/run_number'])
    temp_df['event_number'] = np.array(file[path]['header/event_number'])
    temp_df['trigger_time'] = np.array(file[path]['header/trigger_time'])
    temp_df['radiant_triggers'] = np.array(file[path]['header/trigger_info/trigger_info.radiant_trigger'])
    temp_df['lt_triggers'] = np.array(file[path]['header/trigger_info/trigger_info.lt_trigger'])
    temp_df['force_triggers'] = np.array(file[path]['header/trigger_info/trigger_info.force_trigger'])
    temp_df['ext_triggers'] = np.array(file[path]['header/trigger_info/trigger_info.ext_trigger'])
    
    #combined
    #------------------------------------------------------------------------------------------------------
    reader = readRNOGData()

    reader.begin([f'./{HANDCARRY_DATA}/{filepath}'], overwrite_sampling_rate=3200*units.MHz, apply_baseline_correction='approximate')

    # calculate avg RMS per force trigger event and then get an average for each station, run, channel
    force_trigger_events_in_this_file = temp_df.event_number[temp_df.force_triggers == True]
    avg_RMSs = np.zeros((len(force_trigger_events_in_this_file), 24)) # 2D array with rows for every event and 24 columns for each channels
    for i in range(len(force_trigger_events_in_this_file)): # only look at force trigger events
        avg_RMSs[i] = calculate_avg_RMS(reader.get_event_by_index(force_trigger_events_in_this_file.iloc[i]), station) # row i gets the avg values for all 24 antennas for event i
    avg_RMS = np.mean(avg_RMSs, axis=0)
    
    len_event_number = len(temp_df.event_number)
    l1s = np.zeros((len_event_number, 24))
    amps = np.zeros((len_event_number, 24))
    SNRs = np.zeros((len_event_number, 24))
    RMSs = np.zeros((len_event_number, 24))
    imps = np.zeros((len_event_number, 24))
    max_freqs = np.zeros((len_event_number, 24))
    max_spectrums = np.zeros((len_event_number, 24))
    for i in range(len_event_number):
        l1s[i], amps[i], SNRs[i], RMSs[i], imps[i], max_freqs[i], max_spectrums[i] = calc_l1_amp_SNR(reader.get_event_by_index(temp_df.event_number.iloc[i]), station, avg_RMS)
    '''
    for score, scores in zip(['l1', 'amp', 'SNR', 'imp', 'max_freq', 'max_spectrum'], [l1s, amps, SNRs, RMSs, imps, max_freqs, max_spectrums]):
        temp_df[score] = list(scores)
        temp_df[score] = temp_df[score].apply(lambda x: json.dumps(x.tolist()))
    '''
    temp_df['l1'] = list(l1s)
    temp_df['amp'] = list(amps)
    temp_df['SNR'] = list(SNRs)
    temp_df['imp'] = list(imps)
    temp_df['max_freq'] = list(max_freqs)
    temp_df['max_spectrum'] = list(max_spectrums)

    temp_df['l1'] = temp_df['l1'].apply(lambda x: json.dumps(x.tolist()))
    temp_df['amp'] = temp_df['amp'].apply(lambda x: json.dumps(x.tolist()))
    temp_df['SNR'] = temp_df['SNR'].apply(lambda x: json.dumps(x.tolist()))
    temp_df['imp'] = temp_df['imp'].apply(lambda x: json.dumps(x.tolist()))
    temp_df['max_freq'] = temp_df['max_freq'].apply(lambda x: json.dumps(x.tolist()))
    temp_df['max_spectrum'] = temp_df['max_spectrum'].apply(lambda x: json.dumps(x.tolist()))
    

    write_combined_scores_to_db(df = temp_df[['station_number', 'run_number', 'event_number', 'l1', 'amp', 'SNR', 'imp', 'max_freq', 'max_spectrum']], filename = filename, tablename = 'combined_scores')
    write_combined_scores_to_db(df = pd.DataFrame(avg_RMS), filename = filename, tablename = 'avg_RMS') # kind of don't need this, as we only need the avg_RMS values to calculate the scores that we already have anyways in this case
    
    print(temp_df.head(50))
    return f'finished with {filename}'

def main():
    parser = argparse.ArgumentParser(description="Process two inputs and execute a function.")
    
    parser.add_argument("station", type=int, help="station number")
    parser.add_argument("run", type=int, help="run number")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the existing function with the inputs
    result = process_scores(args.station, args.run)
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()