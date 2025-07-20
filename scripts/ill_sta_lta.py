import obspy
from seismicif.datamod.loading_utils import find_paths

tr_start, tr_stop, te_start, te_stop = 2018, 2020, 2021, 2022
network = "XP"
stations = ["ILL11","ILL12","ILL13","ILL14","ILL15","ILL16","ILL17","ILL18"]


for station in stations:
    channel = "EHZ.D"
    if station == "ILL11": channel = "HHZ.D"

    tr_paths, te_paths = find_paths(network, station, channel, tr_start, tr_stop), find_paths("XP", station, channel, te_start, te_stop)
