# EPA surface-site observation operator for CHEEREIO.
#
# Reads CSV files of daily-mean surface PM observations (e.g. pm_daily_new_YYYYMM_*.csv)
# from the directory given in ens_config under EPAsites_dirs[<species>], and compares them with
# a GEOS-Chem surface-level diagnostic (default: "PM10", in ug/m3).
#
# Relevant CSV columns: Latitude, Longitude, Date Local, and the observed value column
# (default: pm10_ambient). Rows with missing or non-positive values are dropped.
#
# Optional ens_config keys (all keyed by species, e.g. {"PM10": "..."}):
#   EPAsites_obs_column : CSV column holding observations   (default "pm10_ambient" for PM10)
#   EPAsites_gc_field   : GEOS-Chem history variable to use (default "PM10_ambient" for PM10, else the species name)

from datetime import datetime, timedelta
from glob import glob
import pickle
import os.path
import re
import numpy as np
import pandas as pd
import observation_operators as obsop

DEFAULT_OBS_COLUMN = {'PM10': 'pm10_ambient'}
DEFAULT_GC_FIELD = {'PM10': 'PM10'} 
MONTH_REGEX = re.compile(r'_(\d{6})_')

def file_month(filename):
	match = MONTH_REGEX.search(os.path.basename(filename))
	if match is None:
		raise ValueError(f'Could not find a YYYYMM month tag in EPA site file name {filename}.')
	return datetime.strptime(match.group(1), "%Y%m")

def next_month(t):
	return datetime(t.year + (t.month // 12), (t.month % 12) + 1, 1)

def read_epasites(filename, obs_column, timeperiod, filterinfo=None):
	df = pd.read_csv(filename, usecols=['Latitude', 'Longitude', 'Date Local', obs_column])
	df = df.dropna()
	df = df[df[obs_column] > 0]
	# Daily means: place each observation at local noon so it falls in the matching 24-hour window.
	df['utctime'] = pd.to_datetime(df['Date Local']) + pd.Timedelta(hours=12)
	df = df[(df['utctime'] >= pd.Timestamp(timeperiod[0])) & (df['utctime'] < pd.Timestamp(timeperiod[1]))]
	if filterinfo is not None and 'MAIN' in filterinfo:
		df = df[np.abs(df['Latitude']) <= filterinfo['MAIN'][0]]
	met = {}
	met['value'] = df[obs_column].values.astype(float)
	met['latitude'] = df['Latitude'].values.astype(float)
	met['longitude'] = df['Longitude'].values.astype(float)
	met['utctime'] = df['utctime'].values.astype('datetime64[ns]')
	return met

class EPAsite_Translator(obsop.Observation_Translator):
	def __init__(self, verbose=1):
		super().__init__(verbose)
	def getObsColumn(self, species):
		col = self.spc_config.get('EPAsites_obs_column', {}).get(species)
		if col is None:
			col = DEFAULT_OBS_COLUMN.get(species)
		if col is None:
			raise ValueError(f'No EPA observation column known for species {species}; set EPAsites_obs_column in ens_config.')
		return col
	def getGCField(self, species):
		return self.spc_config.get('EPAsites_gc_field', {}).get(species, DEFAULT_GC_FIELD.get(species, species))
	# Save dictionary of file months for later use
	def initialReadDate(self):
		sourcedirs = self.spc_config['EPAsites_dirs']
		date_dict = {}
		for key in list(sourcedirs.keys()):
			obs_list = sorted(glob(f'{sourcedirs[key]}/*.csv'))
			date_dict[key] = [file_month(obs) for obs in obs_list]
		with open(f"{self.scratch}/epasites_dates.pickle", 'wb') as handle:
			pickle.dump(date_dict, handle)
		return date_dict
	# Timeperiod is two datetime objects; return files whose month overlaps it
	def globObs(self, species, timeperiod, interval=None):
		sourcedir = self.spc_config['EPAsites_dirs'][species]
		if os.path.exists(f"{self.scratch}/epasites_dates.pickle"):
			with open(f"{self.scratch}/epasites_dates.pickle", 'rb') as handle:
				date_dict = pickle.load(handle)
		else:
			date_dict = self.initialReadDate()
		obs_list = sorted(glob(f'{sourcedir}/*.csv'))
		obs_dates = date_dict[species]
		if len(obs_dates) != len(obs_list): # Files changed since the cache was written
			obs_dates = self.initialReadDate()[species]
		return [obs for obs, t in zip(obs_list, obs_dates) if (t < timeperiod[1]) and (next_month(t) > timeperiod[0])]
	# interval is ignored: observations are daily means.
	def getObservations(self, specieskey, timeperiod, interval=None, includeObsError=False):
		if includeObsError:
			raise ValueError('EPA site observations have no instrument error; set OBS_ERROR_TYPE to "relative" or "absolute".')
		species = self.spc_config['OBSERVED_SPECIES'][specieskey]
		obs_column = self.getObsColumn(species)
		filterinfo = {}
		if specieskey in list(self.spc_config["filter_obs_poleward_of_n_degrees"].keys()):
			filterinfo['MAIN'] = [float(self.spc_config["filter_obs_poleward_of_n_degrees"][specieskey])]
		obs_list = self.globObs(species, timeperiod, interval)
		epa_obs = [read_epasites(obs, obs_column, timeperiod, filterinfo) for obs in obs_list]
		met = {}
		if len(epa_obs) > 0:
			for key in epa_obs[0]:
				met[key] = np.concatenate([obs[key] for obs in epa_obs])
		else:
			met['value'] = np.array([])
			met['latitude'] = np.array([])
			met['longitude'] = np.array([])
			met['utctime'] = np.array([], dtype='datetime64[ns]')
		if self.verbose >= 2:
			print(f'EPAsite_Translator read {len(met["value"])} {species} observations for {timeperiod[0]} to {timeperiod[1]}.')
		return met
	# Nested runs: nearest_loc would snap sites outside the model domain onto edge cells, so drop them.
	# Also drop sites in the nested buffer zone (cells overwritten by boundary conditions); CHEEREIO
	# sets a 3-cell buffer on every side for nested runs. Override with EPAsites_buffer_cells in ens_config.
	def restrictToDomain(self, EPA, GC):
		nbuf = int(self.spc_config.get('EPAsites_buffer_cells', 3))
		lat = GC.lat.values
		lon = GC.lon.values
		if (len(lat) <= 2*nbuf) or (len(lon) <= 2*nbuf):
			raise ValueError(f'Model domain ({len(lat)}x{len(lon)} cells) is too small for a {nbuf}-cell buffer zone.')
		dlat = np.abs(lat[1]-lat[0])
		dlon = np.abs(lon[1]-lon[0])
		inner_lat = lat[nbuf:len(lat)-nbuf]
		inner_lon = lon[nbuf:len(lon)-nbuf]
		keep = (EPA['latitude'] >= inner_lat.min()-dlat/2) & (EPA['latitude'] < inner_lat.max()+dlat/2) \
			& (EPA['longitude'] >= inner_lon.min()-dlon/2) & (EPA['longitude'] < inner_lon.max()+dlon/2)
		if self.verbose >= 2:
			print(f'EPAsite_Translator kept {keep.sum()} of {len(keep)} observations inside the model domain (excluding {nbuf}-cell buffer).')
		return {key: EPA[key][keep] for key in EPA}
	def gcCompare(self, specieskey, EPA, GC, GC_area=None, doErrCalc=True, useObserverError=False, prescribed_error=None, prescribed_error_type=None, transportError=None, errorCorr=None, minError=None):
		species = self.spc_config['OBSERVED_SPECIES'][specieskey]
		gc_field = self.getGCField(species)
		if useObserverError:
			raise ValueError('EPA site observations have no instrument error; set OBS_ERROR_TYPE to "relative" or "absolute".')
		if self.spc_config['NEST'] == 'T':
			EPA = self.restrictToDomain(EPA, GC)
		if len(EPA['value']) == 0:
			return obsop.ObsData(np.array([]), EPA['value'], EPA['latitude'], EPA['longitude'], EPA['utctime'])
		if gc_field not in GC:
			raise ValueError(f'GEOS-Chem field {gc_field} not found in history output. Make sure the collection that saves it is listed in HistoryExtraCollectionsToRead.')
		i, j, t = obsop.nearest_loc(GC, EPA)
		gcfield = GC[gc_field]
		if 'lev' in gcfield.dims:
			gcfield = gcfield.isel(lev=0) # GEOS-Chem Classic output: lev index 0 is the surface
		gcfield = gcfield.transpose('time', 'lat', 'lon')
		GC_on_obs = gcfield.values[t, j, i]
		if self.spc_config['AV_TO_GC_GRID'][specieskey] == "True":
			superObsFunction = self.spc_config['SUPER_OBSERVATION_FUNCTION'][specieskey]
			additional_args_avgGC = {}
			if doErrCalc:
				if prescribed_error is not None:
					additional_args_avgGC['prescribed_error'] = prescribed_error
					additional_args_avgGC['prescribed_error_type'] = prescribed_error_type
				if minError is not None:
					additional_args_avgGC['minError'] = minError
				if errorCorr is not None:
					additional_args_avgGC['errorCorr'] = errorCorr
				if transportError is not None:
					additional_args_avgGC['modelTransportError'] = transportError
			toreturn = obsop.averageByGC(i, j, t, GC, GC_on_obs, EPA['value'], doSuperObs=doErrCalc, superObsFunction=superObsFunction, **additional_args_avgGC)
		else:
			timevals = GC.time.values[t]
			toreturn = obsop.ObsData(GC_on_obs, EPA['value'], EPA['latitude'], EPA['longitude'], timevals)
		return toreturn
