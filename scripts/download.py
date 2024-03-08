import os
from urllib.request import urlretrieve
from zipfile import ZipFile

# specify the output directory and ensuring it exists
OUTPUT_RELATIVE_DIR = "data/landing/"
if not os.path.exists(OUTPUT_RELATIVE_DIR):
    os.makedirs(OUTPUT_RELATIVE_DIR)


def import_taxi_rides_data():
    """
    Referenced from MAST30034 - Prerequisite Notebook
    From the NYC TLC website, download the monthly parquet datasets from
    August 2022 to March 2023 for each type of taxi (Yellow and Green).
    Then, download the dataset containing taxi zone information.
    Store each downloaded file in the data/landing directory.
    """

    # declare the url downloads for each taxi
    YELLOW_URL = \
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_"
    TARGET_DIR = "yellow_taxi"
    TAXI_RELATIVE_DIR = OUTPUT_RELATIVE_DIR + TARGET_DIR

    # specify the year and month for downloads
    YEAR = [2022, 2023]
    MONTHS = [range(8, 13), range(1, 4)]

    # ensuring relevant paths exist or made in the next 2 chunks of if
    # statement and for loop
    if not os.path.exists(OUTPUT_RELATIVE_DIR + TARGET_DIR):
        os.makedirs(OUTPUT_RELATIVE_DIR + TARGET_DIR)
        print(f"Made {TARGET_DIR} folder")

    # for each year, download the relevant parquet file of specified months
    # into taxi type folder
    for i in range(0, len(YEAR)):
        for month in MONTHS[i]:
            print(f"Begin {month}")
            month = str(month).zfill(2)

            url = f"{YELLOW_URL}{YEAR[i]}-{month}.parquet"
            OUTPUT_DIR = f"{TAXI_RELATIVE_DIR}/{YEAR[i]}-{month}.parquet"
            urlretrieve(url, OUTPUT_DIR)
            print(f"Completed {month}")


def import_taxi_zones_data():
    """
    Referenced from MAST30034 - Prerequisite Notebook
    From the NYC TLC website, retrieve and download the CSV and Zip files of
    taxi zone data and geospatial data. Store each of the downloads in the
    "taxi_zones" folder under data/landing directory.
    """

    # declare the url download
    ZONE_SHAPEFILE = \
        "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    TARGET_DIR = "taxi_zones"

    # check if target path exists or being made
    if not os.path.exists(OUTPUT_RELATIVE_DIR + TARGET_DIR):
        os.makedirs(OUTPUT_RELATIVE_DIR + TARGET_DIR)

    # perform downloads for each file type and store in target path
    OUTPUT_DIR_SHAPEFILE = (
        f"{OUTPUT_RELATIVE_DIR + TARGET_DIR}/taxi_zones_shapefile.zip"
    )

    urlretrieve(ZONE_SHAPEFILE, OUTPUT_DIR_SHAPEFILE)

    # extract the shapefile zip file into the directory
    with ZipFile(OUTPUT_DIR_SHAPEFILE, "r") as zip:
        zip.extractall(OUTPUT_RELATIVE_DIR + TARGET_DIR)


def import_subway_data():
    """
    Referenced from MAST30034 - Prerequisite Notebook
    From the NYC Open Data webs`ite, download the dataset of ridership count
    for subway and bus transport in a CSV format, as well as geospatial data
    for subway, bus and express bus routes to the data/landing folder.
    """

    # declare the url download for subway ridership
    RIDERSHIP = "https://data.ny.gov/api/views/wujg-7c2s/rows.csv?accessType=DOWNLOAD&sorting=true"
    TARGET_DIR = "subway"

    # check if target path for subway exists and download relevant file
    # to the target directory
    if not os.path.exists(OUTPUT_RELATIVE_DIR + TARGET_DIR):
        os.makedirs(OUTPUT_RELATIVE_DIR + TARGET_DIR)
        print(f"Made {TARGET_DIR} folder")

    # download ridership data
    OUTPUT_DIR = f"{OUTPUT_RELATIVE_DIR + TARGET_DIR}/subway_ridership.csv"
    urlretrieve(RIDERSHIP, OUTPUT_DIR)


# call all import/download functions
import_taxi_rides_data()
import_taxi_zones_data()
import_subway_data()
