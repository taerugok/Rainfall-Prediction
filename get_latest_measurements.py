
"""import logging
import os
import sys

import numpy as np
import requests

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", logging.INFO))
KNMI_API_KEY=os.environ['knmi_api_key']

class OpenDataAPI:
    def __init__(self, api_token: str):
        self.base_url = "https://api.dataplatform.knmi.nl/open-data/v1"
        self.headers = {"Authorization": api_token}

    def __get_data(self, url, params=None):
        return requests.get(url, headers=self.headers, params=params).json()

    def list_files(self, dataset_name: str, dataset_version: str, params: dict):
        return self.__get_data(
            f"{self.base_url}/datasets/{dataset_name}/versions/{dataset_version}/files",
            params=params,
        )

    def get_file_url(self, dataset_name: str, dataset_version: str, file_name: str):
        return self.__get_data(
            f"{self.base_url}/datasets/{dataset_name}/versions/{dataset_version}/files/{file_name}/url"
        )


def download_file_from_temporary_download_url(download_url, filename):
    try:
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception:
        logger.exception("Unable to download file using download URL")
        sys.exit(1)

    logger.info(f"Successfully downloaded dataset file to {filename}")


api_key=KNMI_API_KEY

dataset_name = "Actuele10mindataKNMIstations"
dataset_version = "2"
logger.info(f"Fetching latest file of {dataset_name} version {dataset_version}")

api = OpenDataAPI(api_token=api_key)

# sort the files in descending order and only retrieve the first file
params = {"maxKeys": 1, "orderBy": "created", "sorting": "desc"}
response = api.list_files(dataset_name, dataset_version, params)
if "error" in response:
    logger.error(f"Unable to retrieve list of files: {response['error']}")
    sys.exit(1)

latest_file = response["files"][0].get("filename")
logger.info(f"Latest file is: {latest_file}")

# fetch the download url and download the file
response = api.get_file_url(dataset_name, dataset_version, latest_file)
download_file_from_temporary_download_url(response["temporaryDownloadUrl"], latest_file)

import xarray as xr
ds = xr.open_dataset(latest_file)
df = ds.to_dataframe()

df=df[df['stationname'].str.contains(r'(?i)eindhoven|rotterdam|amsterdam|utrecht|bosch|maastricht')].copy()
df=df.reset_index(level=[0,1])
df.time=df.time.dt.tz_localize('UTC')
df.time=df.time.dt.tz_convert("Europe/Amsterdam")
df.time=df.time.dt.strftime("%m/%d/%Y, %H:%M:%S %Z")

#data preparation
selected_cols=['station','stationname','time','D1H','R1H','dsd','ff','pp','n','tn','tx']

col_names_dict = {
    "stationname": "stationname",
    "time":"Time",
    "D1H": "Rainfall_Duration_last_hour_minutes",
    "R1H": "Amount_Rainfall_last_Hour_in_mm",
    'dsd':"Wind_Direction",
    "ff":'Wind_Speed_kmh',
    "pp":'Air_pressure_in_hPa',
    "n":'Total_cloud_cover_percentage',
    'tn':'Temperature'
}

#filter out not selected columns
df=df[selected_cols]
df=df[[x for x in df.columns if x in col_names_dict]]

#Convert data to the expected format by LGBM model
df.columns = df.columns.str.strip()
df=df.rename(columns=col_names_dict)
df['Wind_Speed_kmh']=df['Wind_Speed_kmh']*3.6
df['Total_cloud_cover_percentage']=df['Total_cloud_cover_percentage']*100/8
df.loc[df['Total_cloud_cover_percentage'] >= 100, 'Total_cloud_cover_percentage'] = 100
df.loc[df['Total_cloud_cover_percentage'] >= 100, 'Total_cloud_cover_percentage'] = 100
tmp = df.select_dtypes(include=[np.number])
df.loc[:, tmp.columns] = np.round(tmp,2)

#Remove rows with more than 1 NA values
df=df[df.isnull().sum(axis=1)<=1]
//

df=df.fillna('-')

df.to_csv('files/latest_measurements.csv',mode='w',header=True,index=False)
"""
