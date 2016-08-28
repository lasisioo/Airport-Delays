import os
import psycopg2
import urlparse
import sqlalchemy
import pandas as pd
import numpy as np

users = pd.read_csv("../assets1/TeenSquad_UseCase - Users.csv")
interests = pd.read_csv("../assets1/TeenSquad_UseCase - Interests.csv")
localServices = pd.read_csv("../assets1/TeenSquad_UseCase - LocalServices.csv")
civicEngagement = pd.read_csv("../assets1/TeenSquad_UseCase - CivicEngagement.csv")
civicOrganizations = pd.read_csv("../assets1/TeenSquad_UseCase - CivicOrganizations.csv")

localServices["orgzip"] = localServices["orgzip"].fillna(0.0).astype("int", raise_on_error=True)
localServices["userzip"] = localServices["userzip"].fillna(0.0).astype("int", raise_on_error=True)

civicOrganizations["orgzip"] = civicOrganizations["orgzip"].fillna(0.0).astype("int", raise_on_error=True)
civicOrganizations["userzip"] = civicOrganizations["userzip"].fillna(0.0).astype("int", raise_on_error=True)

user = "nosuvhmvbzyrso:V99UPBfry7kvVeJgbWuR_ybU6"
host = "@ec2-54-163-248-14.compute-1.amazonaws.com:5432" 
database = "dc180s339hm2r2"
engine = sqlalchemy.create_engine("postgresql://" + user + host + "/dc180s339hm2r2")

users.to_sql("users",con = engine, if_exists = "replace")
interests.to_sql("interests",con = engine, if_exists = "replace")
localServices.to_sql("localServices",con = engine, if_exists = "replace")
civicEngagement.to_sql("civicEngagement",con = engine, if_exists = "replace")
civicOrganizations.to_sql("civicOrganizations",con = engine, if_exists = "replace")