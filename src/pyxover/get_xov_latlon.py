import pandas as pd


def get_xov_latlon(xov, df):
   """
   Retrieve LAT/LON of xover from geolocalisation table and add to xovers table
   (useful to analyze/plot xovers on map). Updates input xov.xovers.

   :param xov: contains xovers table to merge to
   :param df: mla data table with lat and lon to extract
   """
   # pd.set_option('display.max_columns', 500)

   # only get rows corresponding to former xov location
   df = df.loc[df.seqid_xov == df.seqid_mla][['xovid', 'LON', 'LAT']]
   # will get 2 rows, one for each track in the xover: drop one
   df = df.drop_duplicates(subset=['xovid'], keep='first')
   # assign LAT and LON to their respective xover by id and update
   xov.xovers = pd.merge(xov.xovers, df, left_on='xOvID', right_on='xovid')
   # xov.xovers.drop('xovid',inplace=True)

   return xov