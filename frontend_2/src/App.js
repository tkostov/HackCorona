import React from "react";
import keplerGlReducer from "kepler.gl/reducers";
import { createStore, combineReducers, applyMiddleware } from "redux";
import { taskMiddleware } from "react-palm/tasks";
import { Provider, useDispatch } from "react-redux";
import KeplerGl from "kepler.gl";
import { addDataToMap } from "kepler.gl/actions";
import useSwr from "swr";

const reducers = combineReducers({
  keplerGl: keplerGlReducer
});

const store = createStore(reducers, {}, applyMiddleware(taskMiddleware));

export default function App() {
  return (
    <Provider store={store}>
      <Map />
    </Provider>
  );
}

function Map() {
  const dispatch = useDispatch();
  const { data } = useSwr("covid", async () => {
    const response = await fetch(
      "http://ec2-3-122-224-7.eu-central-1.compute.amazonaws.com:8080/ch_infections"
    );
    const data = await response.json();
    return data;
  });

  React.useEffect(() => {
    if (data) {
      dispatch(
        addDataToMap({
          datasets: {
            info: {
              label: "COVID-19",
              id: "covid19"
            },
            data
          },
          option: {
            centerMap: true,
            readOnly: false
          },
            "config": {
                "visState": {
                    "filters": [
                        {
                            "dataId": [
                                "covid19"
                            ],
                            "id": "qbbcpki7k",
                            "name": [
                                "day"
                            ],
                            "type": "timeRange",
                            "value": [
                                1584837083000,
                                1585340396000
                            ],
                            "enlarged": true,
                            "plotType": "histogram",
                            "yAxis": null
                        }
                    ],
                    "layers": [
                        {
                            "id": "20oz8cf",
                            "type": "heatmap",
                            "config": {
                                "dataId": "covid19",
                                "label": "Point",
                                "color": [
                                    183,
                                    136,
                                    94
                                ],
                                "columns": {
                                    "lat": "latitude",
                                    "lng": "longitude"
                                },
                                "isVisible": true,
                                "visConfig": {
                                    "opacity": 0.8,
                                    "colorRange": {
                                        "name": "Global Warming",
                                        "type": "sequential",
                                        "category": "Uber",
                                        "colors": [
                                            "#5A1846",
                                            "#900C3F",
                                            "#C70039",
                                            "#E3611C",
                                            "#F1920E",
                                            "#FFC300"
                                        ]
                                    },
                                    "radius": 99.6
                                },
                                "textLabel": [
                                    {
                                        "field": null,
                                        "color": [
                                            255,
                                            255,
                                            255
                                        ],
                                        "size": 18,
                                        "offset": [
                                            0,
                                            0
                                        ],
                                        "anchor": "start",
                                        "alignment": "center"
                                    }
                                ]
                            },
                            "visualChannels": {
                                "weightField": null,
                                "weightScale": "linear"
                            }
                        }
                    ],
                    "interactionConfig": {
                        "tooltip": {
                            "fieldsToShow": {
                                "covid19": [
                                    "density",
                                    "day"
                                ]
                            },
                            "enabled": true
                        },
                        "brush": {
                            "size": 0.5,
                            "enabled": false
                        },
                        "coordinate": {
                            "enabled": false
                        }
                    },
                    "layerBlending": "normal",
                    "splitMaps": [],
                    "animationConfig": {
                        "currentTime": null,
                        "speed": 1
                    }
                },
                "mapState": {
                    "bearing": 0,
                    "dragRotate": false,
                    "latitude": 46.57626075046111,
                    "longitude": 7.319057707388097,
                    "pitch": 0,
                    "zoom": 6.958453121881378,
                    "isSplit": false
                },
                "mapStyle": {
                    "styleType": "dark",
                    "topLayerGroups": {},
                    "visibleLayerGroups": {
                        "label": true,
                        "road": true,
                        "border": false,
                        "building": true,
                        "water": true,
                        "land": true,
                        "3d building": false
                    },
                    "threeDBuildingColor": [
                        224.4071295378559,
                        224.4071295378559,
                        224.4071295378559
                    ],
                    "mapStyles": {}
                }
            }
        })
      );
    }
  }, [dispatch, data]);

  return (
    <KeplerGl
      id="covid"
      mapboxApiAccessToken={process.env.REACT_APP_MAPBOX_API}
      width={window.innerWidth}
      height={window.innerHeight}
    />
  );
}
