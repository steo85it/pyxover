#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

def process_rdr(folder, rdr):
    '''Loads selected RDR from the .DAT file as dataframe
       Uses .FMT to scale all variables and replace missing values with np.nan'''

    # Data types from the .FMT file
    # Note that transmit_time is given as two integers in the .DAT files, which I have named
    # transmit_time and transmit_time_sub for now

    dt = np.dtype([ ('met_seconds', 'int32'), ('subseconds', 'uint32'),
                   ('transmit_time', 'uint32'), ('transmit_time_sub', 'uint32'),
                   ('laser_energy', 'int32'),('transmit_width', 'int32'),('sc_longitude', 'int32'),
                   ('sc_latitude', 'int32'),('sc_radius', 'uint32'),('selenoid_radius', 'uint32'),

                   ('longitude_1', 'int32'), ('latitude_1', 'int32'), ('radius_1', 'int32'), ('range_1', 'uint32'),
                   ('pulse_1', 'int32'), ('energy_1', 'uint32'), ('background_1', 'uint32'), ('threshold_1', 'uint32'),
                   ('gain_1', 'uint32'), ('shot_flag_1', 'uint32'),

                   ('longitude_2', 'int32'), ('latitude_2', 'int32'), ('radius_2', 'int32'), ('range_2', 'uint32'),
                   ('pulse_2', 'int32'), ('energy_2', 'uint32'), ('background_2', 'uint32'), ('threshold_2', 'uint32'),
                   ('gain_2', 'uint32'), ('shot_flag_2', 'uint32'),

                   ('longitude_3', 'int32'), ('latitude_3', 'int32'), ('radius_3', 'int32'), ('range_3', 'uint32'),
                   ('pulse_3', 'int32'), ('energy_3', 'uint32'), ('background_3', 'uint32'), ('threshold_3', 'uint32'),
                   ('gain_3', 'uint32'), ('shot_flag_3', 'uint32'),

                   ('longitude_4', 'int32'), ('latitude_4', 'int32'), ('radius_4', 'int32'), ('range_4', 'uint32'),
                   ('pulse_4', 'int32'), ('energy_4', 'uint32'), ('background_4', 'uint32'), ('threshold_4', 'uint32'),
                   ('gain_4', 'uint32'), ('shot_flag_4', 'uint32'),

                   ('longitude_5', 'int32'), ('latitude_5', 'int32'), ('radius_5', 'int32'), ('range_5', 'uint32'),
                   ('pulse_5', 'int32'), ('energy_5', 'uint32'), ('background_5', 'uint32'), ('threshold_5', 'uint32'),
                   ('gain_5', 'uint32'), ('shot_flag_5', 'uint32'),

                   ('offnadir_angle', 'uint16'), ('emission_angle', 'uint16'),
                   ('solar_incidence', 'uint16'), ('solar_phase', 'uint16'),
                   ('earth_range', 'uint32'), ('earth_pulse', 'uint16'), ('earth_energy', 'uint16')
                  ])

    # Missing values from the .FMT file - better in Python to replace all with NaN

    cols_with_missing_constant = {'met_seconds': {-1: np.nan}, 'laser_energy': {-1: np.nan},
                              'transmit_time_sub': {-1: np.nan},'transmit_width': {-1: np.nan},
                              'sc_longitude': {-2147483648: np.nan}, 'sc_latitude': {-2147483648: np.nan},
                              'sc_radius': {4294967295: np.nan}, 'selenoid_radius': {4294967295: np.nan},

                              'longitude_1': {-2147483648: np.nan}, 'latitude_1': {-2147483648: np.nan},
                              'radius_1': {-1: np.nan}, 'range_1': {4294967295: np.nan}, 'pulse_1': {-1: np.nan},

                              'longitude_2': {-2147483648: np.nan}, 'latitude_2': {-2147483648: np.nan},
                              'radius_2': {-1: np.nan}, 'range_2': {4294967295: np.nan}, 'pulse_2': {-1: np.nan},

                              'longitude_3': {-2147483648: np.nan}, 'latitude_3': {-2147483648: np.nan},
                              'radius_3': {-1: np.nan}, 'range_3': {4294967295: np.nan}, 'pulse_3': {-1: np.nan},

                              'longitude_4': {-2147483648: np.nan}, 'latitude_4': {-2147483648: np.nan},
                              'radius_4': {-1: np.nan}, 'range_4': {4294967295: np.nan}, 'pulse_4': {-1: np.nan},

                              'longitude_5': {-2147483648: np.nan}, 'latitude_5': {-2147483648: np.nan},
                              'radius_5': {-1: np.nan}, 'range_5': {4294967295: np.nan}, 'pulse_5': {-1: np.nan},

                              'offnadir_angle': {65535: np.nan}, 'emission_angle': {65535: np.nan},
                              'solar_incidence': {65535: np.nan}, 'solar_phase': {65535: np.nan},
                              'earth_pulse': {65535: np.nan}, 'earth_energy': {65535: np.nan}
                             }


    # Scaling from .FMT file to get actual values - note conversion from radians to degrees

    cols_with_scale = {'subseconds': 1/4294967296.0, 'sc_longitude': 1e-7, 'sc_latitude': 1e-7,
                       'laser_energy': 1e-9, 'transmit_time_sub': 1/4294967296.0, 'transmit_width': 1e-12,
                       'sc_radius': 1e-3, 'selenoid_radius': 1e-3,

                       'longitude_1': 1e-7, 'latitude_1': 1e-7, 'radius_1': 1e-3, 'range_1': 1e-3,
                       'pulse_1': 1e-12, 'energy_1': 1e-21, 'threshold_1': 1e-6, 'gain_1': 1e-6,

                       'longitude_2': 1e-7, 'latitude_2': 1e-7, 'radius_2': 1e-3, 'range_2': 1e-3,
                       'pulse_2': 1e-12, 'energy_2': 1e-21, 'threshold_2': 1e-6, 'gain_2': 1e-6,

                       'longitude_3': 1e-7, 'latitude_3': 1e-7, 'radius_3': 1e-3, 'range_3': 1e-3,
                       'pulse_3': 1e-12, 'energy_3': 1e-21, 'threshold_3': 1e-6, 'gain_3': 1e-6,

                       'longitude_4': 1e-7, 'latitude_4': 1e-7, 'radius_4': 1e-3, 'range_4': 1e-3,
                       'pulse_4': 1e-12, 'energy_4': 1e-21, 'threshold_4': 1e-6, 'gain_4': 1e-6,

                       'longitude_5': 1e-7, 'latitude_5': 1e-7, 'radius_5': 1e-3, 'range_5': 1e-3,
                       'pulse_5': 1e-12, 'energy_5': 1e-21, 'threshold_5': 1e-6, 'gain_5': 1e-6,

                       'offnadir_angle': 5e-5 * 180/np.pi, 'emission_angle': 5e-5 * 180/np.pi,
                       'solar_incidence': 5e-5 * 180/np.pi, 'solar_phase': 5e-5 * 180/np.pi
                      }


    df = pd.DataFrame.from_records(np.fromfile(folder + rdr + '.DAT', dtype=dt, count=-1))
    df.insert(0, 'rdr_name', rdr)

    df.replace(cols_with_missing_constant, inplace=True)

    for key, value in cols_with_scale.items():
        df[key] = np.multiply(df[key], value)

    # transmit_time_sub already scaled
    df['transmit_time'] += df['transmit_time_sub']

    # This is from Mike's matlab script
    for flag in ['shot_flag_1', 'shot_flag_2', 'shot_flag_3', 'shot_flag_4', 'shot_flag_5']:
        df[flag] = np.mod(df[flag], 64)
        df.loc[np.mod(df[flag], 65536) >= 32768, flag] = 15

    df.drop(['transmit_time_sub'], axis='columns', inplace=True)


    return df
