import numpy as np
import matplotlib.pyplot as plt
import NuRadioReco.modules.io.RNO_G.readRNOGDataMattak
from NuRadioReco.utilities import units, fft
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('station', type=int)
parser.add_argument('run', type=int)
parser.add_argument('--number_of_runs', type=int, default=1, help='How many runs are used (default = 1)? The "run" will be used as the first run.')
parser.add_argument('--n_events', type=int, default=1000000)
parser.add_argument('--events_per_block', type=int, default=100)
args = parser.parse_args()
sampling_rate = 3.2
channel_ids = np.arange(24, dtype=int)
n_cols = 4
n_rows = channel_ids.shape[0] // n_cols

readRNOGData = NuRadioReco.modules.io.RNO_G.readRNOGDataMattak.readRNOGData()

filename = []
for run_i in range(args.number_of_runs):
    filename.append('/home/software/Flights/combined/station{}_run{}_combined.root'.format(args.station, args.run + run_i))

print(filename)

readRNOGData.begin(filename, mattak_kwargs={'backend':'uproot'}, overwrite_sampling_rate=3.2*units.GHz, convert_to_voltage=False, apply_baseline_correction=False)

if channel_ids.shape[0] % n_cols > 0:
    n_rows += 1

n_events = min(args.n_events, len(readRNOGData.get_events_information()))
block_size = args.events_per_block
n_blocks = int(np.ceil(n_events / block_size))
connection_points = np.array([32, 64, 128])

connection_jumps = np.zeros((connection_points.shape[0], channel_ids.shape[0], n_blocks, block_size * (2048 // 128 - 1)))
print(connection_jumps.shape)
connection_jumps[:] = np.nan


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / sigma**2) / 2

i_block = 0

for i_event, event in enumerate(readRNOGData.run()):
    print(i_event, end='\r')
    station = event.get_station(args.station)
    if i_event >= n_events:
        break
    j_event = i_event - i_block * block_size
    if j_event == block_size:
        i_block += 1
        j_event = 0
    for i_channel, channel_id in enumerate(channel_ids):
        channel = station.get_channel(channel_id)
        trace = channel.get_trace()
        for i_chunk in range(2048 // 128 - 1):
            for i_connection, connection_point in enumerate(connection_points):
                connection_jumps[i_connection, i_channel, i_block, j_event * (2048 // 128 - 1) + i_chunk] = (trace[i_chunk * 128 + connection_point - 1] - trace[i_chunk * 128 + connection_point])

distribution_widths = np.zeros((connection_points.shape[0], channel_ids.shape[0], n_blocks))
for i_connection in range(connection_points.shape[0]):
    for i_channel in range(channel_ids.shape[0]):
        for i_block in range(n_blocks):
            distribution_widths[i_connection, i_channel, i_block] = np.sqrt(np.nanmean((connection_jumps[i_connection, i_channel, i_block] - np.nanmean(connection_jumps[i_connection, i_channel, i_block]))**2))
fig1 = plt.figure(figsize=(6 * n_cols, 3 * n_rows))
for i_channel, channel_id in enumerate(channel_ids):
    ax1_1 = fig1.add_subplot(n_rows, n_cols, i_channel + 1)
    for i_connection, connection_point in enumerate(connection_points):
        ax1_1.plot(
            np.arange(n_blocks) * block_size + .5 * block_size,
            distribution_widths[i_connection, i_channel],
            label='$\Delta$n={}'.format(connection_point)
        )
    ax1_1.set_title('Channel {}'.format(channel_id))
    ax1_1.grid()
    ax1_1.legend()
    ax1_1.set_xlabel('N block')
    ax1_1.set_ylabel('$\delta$U [a.u.]')
fig1.tight_layout()
plt.show()