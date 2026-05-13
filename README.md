# Seismic Isolation Forest

This repository contains code for applying the isolation forest (IF) trigger to search for mass movements in seismic data, the idea being that mass movements will likely manifest as a significant anomaly in the seismic waveforms. As a step towards discriminating between anomalies caused by mass movements and other sources, such as anthropogenic noise and earthquakes, we provide code for measuring dissimilarity between waveform segments using dynamic time warping (DTW). The techniques are condensed into a semi-supervised and unsupervised workflow to explore for mass movements in seismic data.

## Installation

Install pinned development dependencies using:

```
pip install -r requirements.txt
```

If you are using Conda to manage your Python environments:

```
conda env create -f environment.yml
```

Alternatively, if you are using an existing environment, you can install the module in [editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html), which includes only minimal dependencies:

```
pip install -e .
```

## Repository Structure

* /catalogues - contains catalogues for the Illgraben and Greenland seismic networks
* /data - used to store miniseed recordings. Structure should be /data/network_name/year/station/channel
* /notebooks - notebooks for lightweight illustration of source code
* /output - storage for outputs generated
* /scripts - python scripts
* /src - core codebase


## Getting Started

### Data and catalogs

The miniseed recordings containing the seismic waveforms should be stored in the /data folder. For a chosen network, year, station and channel use the folder:

/data/network_name/year/station/channel

For example in the illustrations we use data obtained from the Illgraben seismic network in 2018 for station ILL18 and channel EHZ.D. Correspondingly we use the folder:

"/data/XP/2018/ILL18/EHZ.D"

where XP is used to refer to Illgraben. The network_name is used to refer to the corresponding network in the Python scripts.

A catalog of events for a given network should be stored in /catalogs/network. In the case of the semi-supervised mining strategy there should be an
initial_catalog.csv and calibration_catalog.csv. The repo contains examples in /catalogs/XP and /catalogs/DK for the Illgraben and Greenland networks.


### Notebooks

To familiarize yourself with the code base we recommend working through the illustrative notebooks, in particular:
 * if_illustration.ipynb: How the IF trigger can be trained and used to extract anomalous waveform segments from seismic data.
 * dtw_illustration.ipynb: How DTW can be used to analyze the IF segments.
 * sta_lta_illustration.ipynb: How the STA-LTA trigger can be used to extract waveform segments from seismic data.

Additionally the following notebooks exist:
 * if_tests.ipynb: Illustrates concepts related to the isolation forest discussed in Sect. 2.2.1 of the paper.
 * karat_clustering.ipynb: Notebook for the clustering of the IF segments obtained from KARAT.
 * plots.ipynb: Notebook generating plots for the paper.

### Scripts

The repository contains several python scripts to generate output. The scripts themselves are built to flag missing folders and make suggestions where to create them.

To generate output for a given station in terms of the IF trigger the following should be run in order from the scripts/ directory:
 * run_if.py: Train an isolation forest to seismic waveforms from a given station and compute time series of the corresponding anomaly scores.
 * calibrate_if_trigger.py: Calibrate the IF trigger thresholds for a given station.
 * get_if_segments.py: Extracts IF segments from time series of IF anomaly scores for a given station.

To generate output for a given station in terms of DTW the following should be run in order from the scripts/ directory:
 * get_ref_segments.py: Extracts reference segments for subsequent use in segment DTW for a given station.
 * run_dtw.py: Performs Dynamic Time Warping (DTW) between segments and reference segments.

To generate output for a given station in terms of the classical STA-LTA trigger the following should be run in order from the scripts/ directory:
 * calibrate_slta.py: Calibrate the Short-Term Average over Long-Term Average STA-LTA trigger parameters.
 * get_slta_segments.py: Extracts STA-LTA segments from miniseed recordings.

In addition the following scripts are relevant:
 * get_detections.py: Generate detections according to the semi-supervised workflow.
 * add_control_scores.py: Add time series of IF anomaly scores from a control station to time series of IF anomaly scores from a target station.
