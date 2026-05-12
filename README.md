# Seismic Isolation Forest

This repository contains code for applying the isolation forest trigger to search for mass movements in seismic data, the idea being that mass movements will likely manifest as a significant anomaly in the seismic waveforms. To discriminate between anomalies caused by mass movements and other sources, such as anthropogenic noise and earthquakes, we provide code for measuring dissimilarity between segments of the seismic waveforms using dynamic time warping (DTW).  Additionally, we provide code for comparing the classical sta-lta trigger to its isolation forest counterpart.


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
* /data - used to store miniseed recordings. Structure should be /data/network/year/station/channel
* /notebooks - notebooks for lightweight illustration of source code
* /output - storage for outputs generated
* /scripts - python scripts
* /src - core codebase

## Getting Started
