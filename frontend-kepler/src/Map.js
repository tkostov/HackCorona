import React from "react";
import {Provider, useDispatch} from "react-redux";
import KeplerGl from "kepler.gl";
import {addDataToMap} from "kepler.gl/actions";
import useSwr from "swr";
import {createStore, combineReducers, applyMiddleware} from "redux";
import keplerGlReducer from "kepler.gl/dist/reducers";
import {taskMiddleware} from "react-palm/tasks/redux";
import {ActionTypes} from 'kepler.gl/actions';
import {handleActions} from 'redux-actions';


const appReducer = handleActions({
    // listen on kepler.gl map update action to store a copy of viewport in app state
    [ActionTypes.LAYER_CLICK]: (state, action) => {
        console.log('logging vis state', state, action);
        return action;
    }
}, {});

const reducers = combineReducers({
    app: appReducer,
    keplerGl: keplerGlReducer
});

const store = createStore(reducers, {}, applyMiddleware(taskMiddleware));

export default function Map() {
    return (
        < Provider
    store = {store} >
        < MyMap / >
        < /Provider>
)
    ;
}
function MyMap() {
    const dispatch = useDispatch();
    const {data} = useSwr("covid", async () => {
        const response = await fetch(
            "http://ec2-3-122-224-7.eu-central-1.compute.amazonaws.com:8080/data/sqrt"
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
                            }
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
