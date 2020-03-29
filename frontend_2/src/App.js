import React from "react";
import keplerGlReducer from "kepler.gl/reducers";
import {createStore, combineReducers, applyMiddleware} from "redux";
import {taskMiddleware} from "react-palm/tasks";
import {Provider, useDispatch} from "react-redux";
import KeplerGl from "kepler.gl";
import {addDataToMap} from "kepler.gl/actions";
import useSwr from "swr";
import Modal from 'react-modal';

const customStyles = {
  content : {
    top                   : '50%',
    left                  : '50%',
    right                 : 'auto',
    bottom                : 'auto',
    marginRight           : '-50%',
    transform             : 'translate(-50%, -50%)'
  }
};

const customizedKeplerGlReducer = keplerGlReducer
    .initialState({
        uiState: {
            // hide side panel to disallow user customize the map
            readOnly: false,
        }
    });

const reducers = combineReducers({
    keplerGl: customizedKeplerGlReducer,
});

const store = createStore(reducers, {}, applyMiddleware(taskMiddleware));

export default function App() {
    return (
        < Provider
    store = {store} >
        < Map / >
        <Modal
            style={customStyles}
            contentLabel="Example Modal"
        />
        < /Provider>
)
    ;
}

function Map() {
    const dispatch = useDispatch();
    const {data} = useSwr("covid", async () => {
        const response = await fetch(
            "http://ec2-3-122-224-7.eu-central-1.compute.amazonaws.com:8080/infections_sqrt"
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
                        readOnly: true
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
                                        1583020800000,
                                        1583107199000
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
                                        "weightField": {
                                            "name": "cases",
                                            "type": "integer"
                                        },
                                        "columns": {
                                            "lat": "latitude",
                                            "lng": "longitude"
                                        },
                                        "isVisible": true,
                                        "visConfig": {
                                            "colorRange": {
                                                "name": "Global Warming",
                                                "type": "sequential",
                                                "category": "Uber",
                                                "colors": [
                                                    "#E6FAFA",
                                                    "#C1E5E6",
                                                    "#9DD0D4",
                                                    "#75BBC1",
                                                    "#4BA7AF",
                                                    "#00939C"
                                                ]
                                            },
                                            "radius": 100,
                                            "opacity": 0.99,
                                            "field": {
                                                "name": "density",
                                                "type": "integer"
                                            }
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
                                "border": true,
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
        < KeplerGl
    id = "covid"
    mapboxApiAccessToken = {process.env.REACT_APP_MAPBOX_API}
    width = {window.innerWidth}
    height = {window.innerHeight}
    />
)
    ;
}
