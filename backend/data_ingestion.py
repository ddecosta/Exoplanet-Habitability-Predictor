#Imports
import pandas as pd
import numpy as np
import urllib.parse

#Setup data ingestion query
table = "pscomppars"
columns = "pl_name, hostname, discoverymethod, disc_year, pl_orbper, pl_rade, pl_bmasse, st_teff, pl_orbsmax, pl_eqt, st_rad, st_lum, sy_dist, st_met, st_logg, sy_pnum, st_age, st_rotp, pl_orbincl, pl_orbeccen, pl_imppar, st_spectype, st_mass, st_dens, st_vsin, pl_dens, pl_trandep"
query = f"SELECT {columns} FROM {table}"

#Encode URL
base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
full_url = base_url + urllib.parse.quote(query) + "&format=csv"

#Ingest data
print("Fetching data from NASA...")
df = pd.read_csv(full_url)

#Save data to local file
df.to_csv("exoplanet_data.csv", index=False)
print(f"Downloaded {len(df)} planets.")

#Display dataset information
df.head()
df.info()