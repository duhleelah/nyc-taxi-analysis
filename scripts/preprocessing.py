from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, to_timestamp, year, month, \
    dayofweek, hour, dayofmonth, count, sum
from pyspark.sql.types import IntegerType, BooleanType, StringType,\
    StructField, StructType, TimestampNTZType
import os
import json
import warnings
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

warnings.filterwarnings("ignore")


# year and months of taxi data set
YEAR = [2022, 2023]
months = [range(8, 13), range(1, 4)]

# relative paths of raw and curated directory
OUTPUT_RAW_DIR = "data/raw/"
OUTPUT_CURATED_DIR = "data/curated/"

# folders to store preprocessed data
folders = ["subway", "taxi_zones", "yellow_taxi"]

# create relevant folders in data/raw/ directory like in landing directory
for folder in folders:
    if not os.path.exists(OUTPUT_RAW_DIR + folder):
        os.makedirs(OUTPUT_RAW_DIR + folder)
        print(f"Made {folder} folder")

# create relevant folders in data/curated/ directory like in landing directory
for folder in folders:
    if not os.path.exists(OUTPUT_CURATED_DIR + folder):
        os.makedirs(OUTPUT_CURATED_DIR + folder)
        print(f"Made {folder} folder")


# Create a spark session for dealing with spark dataframes
spark = (
    SparkSession.builder.appName("MAST30034 Project 1 - Preprocessing")
    .config("spark.sql.repl.eagerEval.enabled", True)
    .config("spark.sql.parquet.cacheMetadata", "true")
    .config("spark.sql.session.timeZone", "Etc/UTC")
    .getOrCreate()
)


def lowercase_names(df, format_type="spark"):
    """
    Referenced from MAST30034 - Tutorial 2 Notebook
    With a given dataset, return the same dataset with updated lowercase
    column names.
    Parameters:
        df (dataframe): the given dataset with column names to update
        format_type (str, optional): a string signalling the type of dataset
        given which defaults to pyspark dataframe.
    Returns:
        df (dataset): the updated dataset with lowercase columns
    """

    # perform the lowercase column name conversion if it is a pyspark dataframe
    if format_type == "spark":
        lowercase_columns = [
            col(col_name).alias(col_name.lower()) for col_name in df.columns
        ]
        return df.select(*lowercase_columns)
    # perform these set of operations if not a pyspark dataframe
    else:
        lowercase_columns = [col.lower() for col in df.columns]
        return df.set_axis(lowercase_columns, axis=1)


def convert_time(df, time_col):
    """
    With a given dataset and time column, convert the time
    column to a timestamp data type.
    Parameters:
        df (dataframe): the given dataset with timestamp column to convert
        time_col (str): the name of the timestamp column in df

    Returns:
        df (dataframe): the updated dataset
    """

    # specify the timestamp format to capture
    TIMESTAMP_FORMAT = "MM/dd/yyyy hh:mm:ss a"

    # convert the time column and update the dataset columns and its names
    df = df.withColumn("conv_timestamp",
                       to_timestamp(time_col, TIMESTAMP_FORMAT))
    df = df.drop(time_col)
    return df.withColumnRenamed("conv_timestamp", time_col)


def update_time_columns(df, column_name):
    """
    With a given dataset and its timestamp columns, extract the features of
    time like year, month, day, and hour.
    Parameters:
        df (dataframe): the dataframe containing time values
        column_name (str): given timestamp column to be extracted

    Returns:
        df (dataframe): the given dataset with updated column additions.
    """

    # change the column names simultaneously
    df = df.withColumns(
        {
            "year": year(col(column_name)),
            "dayofmonth": dayofmonth(col(column_name)),
            "dayofweek": dayofweek(col(column_name)),
            "month": month(col(column_name)),
            "hour": hour(col(column_name)),
        }
    )
    return df


def subway_data_to_raw():
    """
    Clean the CSV of subway hourly ridership by standardising column names,
    selecting relevant columns, and changing to appropriate data types.
    Then, form a separate shapefile for the coordinates of subway stations.
    Store both CSV and shapefile files to the data/raw/subway/ directory.
    """

    # import subway hourly ridership data
    subway = spark.read.option("header", True).csv(
        "data/landing/subway/subway_ridership.csv"
    )

    # standardise column names
    subway = lowercase_names(subway)

    # drop the specified columns
    drop_cols = (
        "station_complex",
        "payment_method",
        "transfers",
        "routes",
        "georeference",
    )
    subway = subway.drop(*drop_cols)

    # change data types of columns to its appropriate format and values
    subway = (
        subway.withColumn("ridership", col("ridership").cast("int"))
        .withColumnRenamed("transit_timestamp", "timestamp")
        .withColumnRenamed("station_complex_id", "station_id")
        .withColumn("latitude", col("latitude").cast("float"))
        .withColumn("longitude", col("longitude").cast("float"))
    )
    subway = convert_time(subway, "timestamp")

    # impute the values of borough codes to full borough names
    subway = subway.withColumn(
        "borough",
        when(subway.borough == "Q", "Queens")
        .when(subway.borough == "BX", "Bronx")
        .when(subway.borough == "M", "Manhattan")
        .when(subway.borough == "BK", "Brooklyn")
        .otherwise(subway.borough),
    )
    # separate coordinates of subway stations in a pandas dataframe
    station_coords = (
        subway.select("station_id", "latitude", "longitude")
        .dropDuplicates()
        .toPandas()
    )

    # combine the latitude and longitude values to form shapely Point data type
    # and store it as a separate file of subway station coordinates
    station_coords["geometry"] = [
        Point(coord)
        for coord in list(zip(station_coords.longitude,
                              station_coords.latitude))
    ]
    station_coords = station_coords.rename(
        columns={"station_complex_id": "station_id"}
    )
    station_coords = gpd.GeoDataFrame(
        station_coords,
        crs="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
    )

    # save station coordinates as shapefile in /data/raw/subway/ directory
    station_coords.to_file(f"{OUTPUT_RAW_DIR}subway/subway_stations.shp")

    # save it to /data/raw/subway/ directory in parquet format
    subway.write.parquet(f"{OUTPUT_RAW_DIR}subway/subway_ridership.parquet")


def taxi_zones_to_raw():
    """
    Perform cleaning on CSV of taxi zones by standardising column names,
    changing to appropriate data types, and removing irrelevant columns.
    Move the file to the data/raw/taxi_zones directory.
    """

    # import the taxi zones shapefile
    zones_shp = gpd.read_file("data/landing/taxi_zones/taxi_zones.shp")

    # standardise column names to lowercase and drop irrelevant columns
    zones_shp = lowercase_names(zones_shp, format_type="pandas")
    keep_cols = ["locationid", "borough", "zone", "geometry"]
    zones_shp = zones_shp[keep_cols]

    # change to appropriate data type for geometry column
    # other columns are strings which is ideal
    zones_shp["geometry"] = zones_shp["geometry"].to_crs(
        "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    )

    # save it to /data/raw/taxi_zones/ directory in shapefile format
    # for further preprocessing later
    zones_shp.to_file(f"{OUTPUT_RAW_DIR}taxi_zones/taxi_zones.shp")


def yellow_taxi_to_raw():
    """
    Referenced from MAST30034 - Tutorial 2 Notebook
    Clean the yellow taxi data from August 2022 to March 2023 by
    standardising column names, changing to appropriate data types,
    and removing irrelevant columns.
    Move the file to the data/raw/yellow_taxi/ directory
    """

    TARGET_DIR = "yellow_taxi"

    # index of columns in schema to change data types from long to int
    int_cols = [3, 5, 9]

    # specify the column to change to boolean data type
    BOOLEAN_COL = 6

    # take the schema from february 2023 as it is closest to final ideal schema
    final_schema = spark.read.parquet(
        "data/landing/yellow_taxi/2023-02.parquet"
        ).schema

    # change the column names to lowercasem while changing
    # a few columns to either int or boolean data types
    for index in range(0, len(final_schema)):
        final_schema[index].name = final_schema[index].name.lower()

        if index in int_cols:
            final_schema[index].dataType = IntegerType()
        elif index == BOOLEAN_COL:
            final_schema[BOOLEAN_COL].dataType = BooleanType()
        else:
            continue

    # loop through monthly parquet files to perform basic cleaning techniques
    # and then append them to a main parquet file
    for i in range(0, len(YEAR)):
        for m in months[i]:
            month = str(m).zfill(2)
            yellow_taxi = spark.read.parquet(
                f"data/landing/yellow_taxi/{YEAR[i]}-{month}.parquet"
            )

            # standardise column names to lowercase
            yellow_taxi = lowercase_names(yellow_taxi)

            # update the data set to follow the decided schema
            yellow_taxi = yellow_taxi.select([
                col(c).cast(final_schema[i].dataType)
                for i, c in enumerate(yellow_taxi.columns)]
            )

            # rename the pickup and dropoff timestamp
            yellow_taxi = yellow_taxi.withColumnsRenamed(
                {"tpep_pickup_datetime": "pu_time",
                 "tpep_dropoff_datetime": "do_time"}
            )

            # save the cleaned form of data set in /data/raw/yellow_taxi/
            # directory as parquet
            yellow_taxi.write.mode("append").parquet(
                f"{OUTPUT_RAW_DIR + TARGET_DIR}/yellow_taxi.parquet"
            )


def filter_taxi_rides(df, interested_areas):
    """
    With a given dataset and a list of location ids, filter the given
    data to meet the business rules and relevant areas of the research.
    Parameters:
        df (dataframe): the dataset dataframe
        interested_areas (list): list of location ids of NYC
    Returns:
        df (dataframe): the filtered dataset
    """

    # specify any constants such as time, distance,
    # and taxi limitations for filtering
    TOTAL_ZONES = 263  # as specified in the TLC Record User Guide
    MIN_THRESH = 0  # boundary of passenger count,
    # trip distance and fee-related amounts
    MAX_FARE = 500  # maximum fare for a customer
    INITIAL_CHARGE = 2.50  # prior to fare changes in December 2022
    MAX_PASSENGER = 6  # maximum legal limit of passengers in a taxi
    MAX_DISTANCE = 50  # longest distance to travel in NYC plus 15
    MAX_TAX = 0.5  # specified MTA tax
    MAX_TOLL = 256.95  # maximum toll paid possible if all highways are visited
    MAX_CATEGORY = 6  # highest value for categorical columns

    # timeline constants for filtering
    YEAR1 = [2022]
    YEAR2 = [2023]
    months_1 = [8, 9, 10, 11, 12]
    months_2 = [1, 2, 3]

    # remove any records beyond the specified years
    df = df.where(
        ((year(col("pu_time")).isin(YEAR1)) &
         (month(col("pu_time")).isin(months_1)))
        | ((year(col("pu_time")).isin(YEAR2)) &
           (month(col("pu_time")).isin(months_2)))
    )

    # since pickups and dropoffs in Staten Island or Newark Airport are not
    # relevant, filter it using the list formed earlier, interested_areas,
    # by keeping relevant locations. concurrently, let's remove any rows
    # that are not within the TLC taxi zone boundaries
    df = df.where(
        (col("pulocationid").isin(interested_areas))
        & (col("dolocationid").isin(interested_areas))
        & (col("pulocationid") <= TOTAL_ZONES)
        & (col("dolocationid") <= TOTAL_ZONES)
    )

    # remove rows that recorded the same pickup and dropoff time
    # or if the dropoff time is earlier than the pickup time
    df = df.where(
        (col("pu_time") != col("do_time")) & (col("pu_time") < col("do_time"))
    )

    # remove any rows with a zero trip distance
    df = df.where(
        (col("trip_distance") > MIN_THRESH)
        & (col("trip_distance") <= MAX_DISTANCE)
        )

    # remove any rows with negative tip amounts and exceed total
    df = df.where(
        (col("tip_amount") >= MIN_THRESH)
        & (col("tip_amount") < col("total_amount"))
        )

    # remove any fare and total amounts that are below
    # initial charge and exceed $1000
    df = df.where(
        (col("fare_amount") >= INITIAL_CHARGE)
        & (col("total_amount") >= INITIAL_CHARGE)
        & (col("fare_amount") < MAX_FARE)
    )

    # remove any tax values that are negative or more than $0.50,
    # toll amount that exceed possible total, and any extras
    # that are negative
    df = df.where(
        (col("mta_tax") <= MAX_TAX)
        & (col("mta_tax") >= MIN_THRESH)
        & (col("tolls_amount") <= MAX_TOLL)
        & (col("tolls_amount") >= MIN_THRESH)
        & (col("extra") >= MIN_THRESH)
    )

    # remove any records with passengers above the lawful limit
    df = df.where(
        (col("passenger_count") <= MAX_PASSENGER)
        & (col("passenger_count") > MIN_THRESH))

    # remove any rows with unlisted values in categories
    df = df.where(
        (col("ratecodeid") <= MAX_CATEGORY)
        & (col("payment_type") <= MAX_CATEGORY)
    )

    return df


def taxi_rides_to_curated():
    """
    Perform preprocessing such as row filtering, data imputation, and data
    aggregation for yellow taxi data. Then, form a final dataframe with total
    yellow taxi pickups and dropoffs per location id for a given day and hour.
    Save this final dataset in the /data/curated/yellow_taxi/ directory.
    """

    TARGET_DIR = "yellow_taxi"  # target directory to save files later

    # columns to be dropped at the end
    drop_cols = (
        "pu_time",
        "do_time",
        "fare_amount",
        "tip_amount",
        "total_amount",
        "trip_distance",
        "vendorid",
        "ratecodeid",
        "store_and_fwd_flag",
        "payment_type",
        "extra",
        "mta_tax",
        "congestion_surcharge",
        "improvement_surcharge",
        "tolls_amount",
        "airport_fee",
    )

    # load the zones shapefile for matching pickup and drop off locations
    zones_shp = gpd.read_file("data/raw/taxi_zones/taxi_zones.shp")

    # load the yellow taxi records
    yellow_taxi = spark.read.parquet(
        f"{OUTPUT_RAW_DIR + TARGET_DIR}/yellow_taxi.parquet"
    )
    yellow_taxi = yellow_taxi.orderBy("pu_time")

    # specify the location ids of not Staten Island and Newark Airport areas
    interested_areas = [
        loc
        for loc in zones_shp[
            zones_shp["borough"].str.contains(
                "Brooklyn|Queens|Bronx|Manhattan")
        ]["locationid"]
    ]

    # filter the zones shapefile to match interested areas
    # and write the geoJSON file into the curated directory
    zones_shp = zones_shp[zones_shp["borough"] != "Staten Island"]
    zones_shp = zones_shp[zones_shp["borough"] != "EWR"]
    zones_shp.to_file(f"{OUTPUT_CURATED_DIR}taxi_zones/taxi_zones.shp")
    zonesJSON = zones_shp.to_json()
    with open(
            f"{OUTPUT_CURATED_DIR}/taxi_zones/zonesJSON.json", "w") as f:
        json.dump(zonesJSON, f)
    # 242 - zones_shp row count after this removal

    yellow_taxi = filter_taxi_rides(yellow_taxi, interested_areas)

    # create new time columns such as year, month, day of the month/week,
    # and hour of the day
    yellow_taxi = update_time_columns(yellow_taxi, "pu_time")

    yellow_taxi = yellow_taxi.drop(*drop_cols)

    # group the taxi trips by locationid and time to determine hourly
    # pickups
    taxi_trips = yellow_taxi.groupBy(
        "pulocationid", "year", "month", "dayofmonth", "dayofweek", "hour")\
        .agg(
            count("hour").alias("pu_hourly"),
            sum("passenger_count").alias("hourly_passengers"))\
        .orderBy("year", "month", "dayofweek", "pulocationid")

    # write the taxi trips into a parquet file and save it in
    # the curated directory
    taxi_trips.write.parquet(
        f"{OUTPUT_CURATED_DIR}{TARGET_DIR}/taxi_trips.parquet")


def subway_rides_to_curated():
    """
    Basically, I want to turn the coordinates of each subway station into a
    point system and then allocate it to a locationid based on the locationid
    polygon shape. This is so that when predicting taxi demand based on the
    number of riders per hour and the number of stations in the area.
    """

    sub_data = spark.read.parquet("data/raw/subway/subway_ridership.parquet")
    sub_stations = gpd.read_file("data/raw/subway/subway_stations.shp")
    zones_shp = gpd.read_file("data/curated/taxi_zones/taxi_zones.shp")

    # initialise a new column in sub_stations to store locationids
    # of where the station is located at
    init_values = pd.Series(i for i in range(0, len(sub_stations)))
    sub_stations = sub_stations.assign(locationid=init_values)

    # find the corresponding locationid for each of the located subway stations
    for i in range(0, len(sub_stations)):
        point = sub_stations["geometry"][i]
        for j in range(0, len(zones_shp)):
            if point.within(zones_shp["geometry"][j]):
                sub_stations["locationid"][i] = zones_shp["locationid"][j]

    # merge the zones shapefile and subway stations into one
    # geopandas df
    station_coords = pd.merge(
        sub_stations, zones_shp, on="locationid", how="left")
    station_coords = station_coords.rename(columns={
        "geometry_x": "station_geometry",
        "geometry_y": "location_geometry"
    })
    stations_and_locations = sub_stations[["station_id", "locationid"]]
    sub_stations.to_file(f"{OUTPUT_CURATED_DIR}subway/station_coords.shp")

    # filter the dates to be within the decided timeline
    sub_data = sub_data.where(
        (col("timestamp") >= "2022-08-01 00:00:00")
        & (col("timestamp") < "2023-04-01 00:00:00")
    )

    # aggregate the data to sum the total hourly ridership by station
    sub_data = sub_data.groupBy("station_id", "timestamp")\
        .agg(sum("ridership").alias("hourly_ridership"))

    # add new time related columns such as year, month, day, and
    # hour
    sub_data = update_time_columns(sub_data, "timestamp")
    sub_data = sub_data.toPandas()

    sub_data = pd.merge(
        sub_data, stations_and_locations,
        on="station_id", how="left"
    )

    # save df as shapefile into curated
    sub_data.to_csv(
        f"{OUTPUT_CURATED_DIR}subway/subway_ridership.csv", index=False)


def combined_taxi_and_subway():
    """
    Combine the hourly taxi pickups and hourly subway riderships into one df
    based on the locationid.
    """

    # specify fill in value for null column values
    FILL_VALUE = 0

    # establish the schema for dataframe to be read in
    subwaySchema = StructType([
        StructField("station_id", StringType(), True),
        StructField("timestamp", TimestampNTZType(), True),
        StructField("hourly_ridership", IntegerType(), True),
        StructField("year", IntegerType(), True),
        StructField("dayofmonth", IntegerType(), True),
        StructField("dayofweek", IntegerType(), True),
        StructField("month", IntegerType(), True),
        StructField("hour", IntegerType(), True),
        StructField("locationid", IntegerType(), True),
    ])

    # load in the subway ridership and taxi rides dataframe
    subway_data = spark.read.format("csv")\
        .option("header", "true")\
        .schema(subwaySchema)\
        .load(f"{OUTPUT_CURATED_DIR}subway/subway_ridership.csv")

    taxi_rides = spark.read.parquet(
        f"{OUTPUT_CURATED_DIR}yellow_taxi/taxi_trips.parquet")

    # group the hourly subway ridership by location and timestamp
    agg_data = subway_data.groupBy("locationid", "timestamp")\
        .agg(sum("hourly_ridership").alias("hourly_ridership"))

    # add new time related columns like year, month, dayofmonth,
    # dayofweek, and hour
    agg_data = update_time_columns(agg_data, "timestamp")

    # with the aggregated subway columns and taxi rides,
    # combine them into a dataframe
    taxi_rides = taxi_rides.toPandas()
    taxi_rides = taxi_rides.rename(columns={"pulocationid": "locationid"})
    agg_data = agg_data.toPandas()

    conds = ["locationid", "year", "month", "dayofmonth", "dayofweek", "hour"]
    subway_taxi_df = pd.merge(
        taxi_rides, agg_data, on=conds, how="outer"
    )

    # fill any null values with 0 as likely the location has no train stations
    subway_taxi_df = subway_taxi_df.drop(["timestamp"], axis=1)
    subway_taxi_df = subway_taxi_df.fillna(int(FILL_VALUE))

    # save as a csv into /data/curated/ directory for visualisations
    subway_taxi_df.to_csv(f"{OUTPUT_CURATED_DIR}visualisation_recs.csv",
                          index=False)

    # separate the dataset into 2022 and 2023 for training and testing
    subway_taxi_2022 = subway_taxi_df.loc[subway_taxi_df["year"] == YEAR[0]]
    subway_taxi_2023 = subway_taxi_df.loc[subway_taxi_df["year"] == YEAR[1]]

    # save the files as csv under curated
    subway_taxi_2022.to_csv(f"{OUTPUT_CURATED_DIR}2022_records.csv",
                            index=False)
    subway_taxi_2023.to_csv(f"{OUTPUT_CURATED_DIR}2023_records.csv",
                            index=False)


# call all the functions to perform basic data transformations for datasets
yellow_taxi_to_raw()
taxi_zones_to_raw()
subway_data_to_raw()

# call all preprocessing functions for each dataset to
# save in curated
taxi_rides_to_curated()
subway_rides_to_curated()
combined_taxi_and_subway()
