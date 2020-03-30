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

# Load R file
r.source('R/code.R')

# (stolen from json_to_pandas.py)
dl = DataLoader()  # instantiate DataLoader
data_dict = dl.process_data()  # loads and forms the data dictionary
rki_data = data_dict["RKI_Data"]  # only RKI dataframe
#print(rki_data.head())

# formatted will contain the data adapted for the R code
# Extract relevant columns
formatted = rki_data[["Meldedatum", "Bundesland", "AnzahlFall"]]

# Rename columns
formatted.rename(columns={
    "Meldedatum": "date",
    "Bundesland": "country",
    "AnzahlFall": "cases"
}, inplace=True)

# NOTE: For testing purposes only to demonstrate that data can be formatted and passed between R and Python
# Filter just three countries and replace them with names the R program knows
formatted = formatted.loc[formatted['country'].isin(['Baden-Württemberg', 'Hamburg', 'Bayern'])]
formatted['country'] = formatted['country'].map({'Baden-Württemberg': 'Guinea', 'Hamburg': 'Liberia', 'Bayern': 'SierraLeone'})

# Sum and group cases by week for each country
formatted['date'] = pd.to_datetime(formatted['date'], unit="ms") - pd.to_timedelta(7, unit='d')
formatted = formatted.groupby([pd.Grouper(key='date', freq='W-MON'), 'country'])['cases'].sum().reset_index()

# Compute cumulative sum of cases for each country
#countries = formatted.country.unique()
#for i, c in enumerate(countries):
#    formatted.loc[formatted['country'] == c, 'cases'] = formatted.loc[formatted['country'] == c, 'cases'].cumsum()

# Sort by country then week
formatted.insert(loc=0, column='week', value=formatted['date'].dt.week)
formatted = formatted.sort_values(by=['country', 'week']).reset_index(drop=True)

# convert to R object
with localconverter(ro.default_converter + pandas2ri.converter):
    dat = ro.conversion.py2rpy(formatted)

# Run R script
# NOTE: code.R is hard-coded to compute forecast for Sierra Leone.
r.init(dat)
forecast = r.run()

# Print results
print('finished with R file. Output: ')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(forecast)

# Example:
#from rpy2.robjects.packages import importr
#utils = importr('utils')
#print(utils.head(dat))
