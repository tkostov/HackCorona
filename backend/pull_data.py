# Unite, if needed
def pull_rki_data():
    import urllib.request, json 
    with urllib.request.urlopen("https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_COVID19/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json") as url:
        data = json.loads(url.read().decode())
        return(data)

def pull_lk_geodata():
    import urllib.request, json 
    with urllib.request.urlopen("https://public.opendatasoft.com/api/records/1.0/search/?dataset=landkreise-in-germany&rows=500&facet=iso&facet=name_0&facet=name_1&facet=name_2&facet=type_2&facet=engtype_2&refine.name_0=Germany") as url:
        data = json.loads(url.read().decode())
        return(data)