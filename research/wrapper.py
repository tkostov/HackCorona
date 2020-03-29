import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri
from json_to_pandas import DataLoader

from rpy2.robjects.conversion import localconverter

# Columns with translations to avoid headaches:

# IdBundesland      | IdFederalstate
# Bundesland        | Federalstate
# Landkreis         | District
# Altersgruppe      | Agegroup
# Geschlecht        | Gender
# AnzahlFall        | NumberofCases
# AnzahlTodesfall   | NumberofDeaths
# ObjectId          | ObjectId
# Meldedatum        | Registrationdate
# IdLandkreis       | IdDistrict
# Bev Insgesamt     | Bev Overall

# Relevant columns from RKI data and the corresponding columns of the ebola model
# (index column) -> week
#   might need incrementing by one
#   (column name semantically incorrect)
# Meldedatum -> date
#   currently in unix time format (in milliseconds?!)
#   needs to be mapped to R Dates
# Bundesland -> country
#   Could use Landkreis instead, ask Todor
#   (column name semantically incorrect)
# AnzahlFall -> cases
#   Not sure if we need further processing, not 100% sure what cases represents in the R file
#   Slightly suspicious because the column looks different (cases consists of decimals, AnzahlFall has small integers)

# (stolen from json_to_pandas.py)
dl = DataLoader()  # instantiate DataLoader
data_dict = dl.process_data()  # loads and forms the data dictionary
rki_data = data_dict["RKI_Data"]  # only RKI dataframe
#print(rki_data.head())

# formatted will contain the data adapted for the R code
# Extract relevant columns
formatted = rki_data[["Meldedatum", "Bundesland", "AnzahlFall"]]

# Add the week column
formatted.insert(loc=0, column="week", value=(formatted.index+1))

# Rename columns
formatted.rename(columns={
    "Meldedatum": "date",
    "Bundesland": "country",
    "AnzahlFall": "cases"
}, inplace=True)

# Convert from UNIX format to datetimes, might want to format further
# TODO: check if this accurately converts, e.g. accounts for calendrical oddities
formatted["date"] = pd.to_datetime(formatted["date"], unit="ms")

# convert to R object
with localconverter(ro.default_converter + pandas2ri.converter):
    dat = ro.conversion.py2rpy(formatted)

# Example:
from rpy2.robjects.packages import importr
utils = importr('utils')
print(utils.head(dat))
