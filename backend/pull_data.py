# Unite, if needed
def pull_rki_data():
    import urllib.request, json 
    from datetime import timedelta, date

    def daterange(start_date, end_date):
        for n in range(int ((end_date - start_date).days)):
            yield start_date + timedelta(n)

    start_date = date(2020, 1, 1)
    end_date = date.today() + timedelta(days=1)
    list_data = []
    for single_date in daterange(start_date, end_date):
        with urllib.request.urlopen("https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_COVID19/FeatureServer/0//query?where=Meldedatum%3D%27"+str(single_date.strftime("%Y-%m-%d"))+"%27&objectIds=&time=&resultType=none&outFields=*&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token=") as url:
            json_data = json.loads(url.read().decode())
            list_data.append(json_data)
    return(list_data)

def pull_lk_geodata():
    import urllib.request, json 
    with urllib.request.urlopen("https://public.opendatasoft.com/api/records/1.0/search/?dataset=landkreise-in-germany&rows=500&facet=iso&facet=name_0&facet=name_1&facet=name_2&facet=type_2&facet=engtype_2&refine.name_0=Germany") as url:
        data = json.loads(url.read().decode())
        return(data)
