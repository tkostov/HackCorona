# Data Sources of Track-The-Virus
The following data sources were processed to generate our own database. Our database does not include all the data that is collected. Also calculations of these data were made to get the desired values.

# Germany
### Covid-19 Cases
[Example source for 27.03.2020](https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_COVID19/FeatureServer/0//query?where=Meldedatum%3D%272020-03-28%27&objectIds=&time=&resultType=none&outFields=*&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token=)

| Field Name          | Description                                |
|---------------------|--------------------------------------------|
| IdBundesland | ID for Region|
| Bundesland  | Region|
| Landkreis  | Municipale|
| Altersgruppe  | Age group|
| Geschlecht  | Sex|
| AnzahlFall  | Amount of cases|
| AnzahlTodesfall  | Amount of deaths|
| ObjectId  | Object ID|
| Meldedatum  | Date of notification|
| IdLandkreis  | Municipale ID|
| Datenstand  | Data status|
| NeuerFall  | New case |
| NeuerTodesfall | New death|

### Demographics
- [Source](https://public.opendatasoft.com/api/records/1.0/search/?dataset=landkreise-in-germany&rows=500&facet=iso&facet=name_0&facet=name_1&facet=name_2&facet=type_2&facet=engtype_2&refine.name_0=Germany)
- `data/bev_lk.xlsx` Source ???

# Switzerland
### Covid-19 Cases
[Example source for Zurich](https://raw.githubusercontent.com/openZH/covid_19/master/fallzahlen_kanton_total_csv/COVID19_Fallzahlen_Kanton_ZH_total.csv)

All sources can be found in `/covid19-cases-switzerland/sources.ini`

| Field Name          | Description                                |
|---------------------|--------------------------------------------|
| date               | Date of notification                       |
| time                | Time of notification                       |
| abbreviation_canton_and_fl | Abbreviation of the reporting canton       | 
| ncumul_tested      | Reported number of tests performed as of date|
| ncumul_conf          | Reported number of confirmed cases as of date|
| ncumul_hosp         | Reported number of hospitalised patients on date|
| ncumul_ICU          | Reported number of hospitalised patients in ICUs on date| 
| ncumul_vent         | Reported number of patients requiring ventilation on date | 
| ncumul_released     |Reported number of patients released from hospitals or reported recovered as of date|
| ncumul_deceased     |Reported number of deceased as of date|
| source              | Source of the information                  |

### Demographics
???

# Italy
### Covid-19 Cases
[Source](https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv)

| Data                  | Description                            |
|-----------------------------|----------------------------------------|
| data                        | Date of notification                   | 
| stato                       | Country of reference                   | 
| codice_regione              | Code of the Region (ISTAT 2019)        | 
| denominazione_regione       | Name of the Region                     | 
| lat                         | Latitude                               | 
| long                        | Longitude                              | 
| ricoverati_con_sintomi      | Hospitalised patients with symptoms    | 
| terapia_intensiva           | Intensive Care                         | 
| totale_ospedalizzati        | Total hospitalised patients            | 
| isolamento_domiciliare      | Home confinement                       | 
| totale_attualmente_positivi | Total amount of current positive cases (Hospitalised patients + Home confinement)  |
| nuovi_attualmente_positivi  | News amount of current positive cases (Actual total amount of current positive cases - total amount of current positive cases of the previous day)| 
| dimessi_guariti             | Recovered                              | 
| deceduti                    | Death                                  |
| totale_casi                 | Total amount of positive cases         |
| tamponi                     | Tests performed                        | 



### Demographics 

[Source](https://www.tuttitalia.it/regioni/)


|  Data   | Description |
|---------|----------------------------------------|
| Regione  |  Region |
| Popolazione  |  Population |
| Superficie  |  Area |
| Densità  |  Density|
| Numero Comuni  | Number of municipalities  |
| Numero Province  |  Number of provinces |
