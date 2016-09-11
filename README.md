# Airport Delays

## Requirements: Python 2.7, Anaconda with following python libraries installed

1. numpy
2. pandas
3. psycopg2
4. matplotlib
5. plotly
6. sqlalchemy
7. sklearn


## Applications used:
- KMeans Clustering
- Principal Compnent Analysis

<b>Problem Statement:</b> We want to understand the behaviour of individual airport operations that lead to cancellations and delays in those airports.

## Data Dictionary:

| Variable | Description | Data Type | Variable Type
| --- | --- | :---: | --- |
| airport | The unique number to represent an airport | Integer | Unique |
| year | The year related to the airport| Integer | Continuous |
| departure cancellations | The number of departure cancellations in an airport in a given year | Integer | Continuous |
| arrival cancellations | The number of arrival cancellations in an airport in a given year | Integer | Continuous |
| departure diversions | The number of departures that were diverted in an airport in a given year | Integer | Continuous |
| arrival diversions | The number of arrivals that were diverted in an airport in a given year | Integer | Continuous |
| percent on-time gate departures | The percentage of on-time gate departures in an airport in a given year | Float | Continuous |
| percent on-time airport departures | The percentage of on-time airport departures in an airport in a given year | Float | Continuous |
| percent on-time gate arrivals | The percentage of on-time gate arrivals in an airport in a given year | Float | Continuous |
| average_gate_departure_delay | The average daily number of gate departure delays in an airport in a given year, in minutes | Float | Continuous |
| average_taxi_out_time	| The average taxi-out time in an airport in a given year, in minutes | Float | Continuous |
| average taxi out delay | The average taxi-out delay in an airport in a given year, in minutes | Float | Continuous |
| average airport departure delay | The average airpoty delay on arrival in an airport in a given year, in minutes | Float | Continuous |
| average airborne delay | The difference between actual and scheduled airborne time, in minutes | Float | Continuous |
| average taxi in delay	| The average taxi-out delay in an airport in a given year, in minutes | Float | Continuous |
| average block delay | The difference between actual and scheduled gate-to-gate time, in minutes | Float | Continuous |
| average gate arrival delay | The average gate delay on arrival in an airport in a given year, in minutes | Float | Continuous |
