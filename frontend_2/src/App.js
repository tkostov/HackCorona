import React from "react";
import keplerGlReducer from "kepler.gl/reducers";
import {createStore, combineReducers, applyMiddleware} from "redux";
import {taskMiddleware} from "react-palm/tasks";
import {Provider, useDispatch} from "react-redux";
import KeplerGl from "kepler.gl";
import {addDataToMap} from "kepler.gl/actions";
import useSwr from "swr";

const reducers = combineReducers({
    keplerGl: keplerGlReducer
});

const store = createStore(reducers, {}, applyMiddleware(taskMiddleware));

export default function App() {
    return (
        < Provider
    store = {store} >
        < Map / >
        < /Provider>
)
    ;
}

function Map() {
    const dispatch = useDispatch();
    const {data} = useSwr("covid", async () => {
        const response = await fetch(
            "http://localhost:8080/infections"
        );
        const data = await response.json();
        return data;
    });

    const sampleConfig = {
        visState: {
            filters: [
                {
                    id: 'me',
                    dataId: 'covid19',
                    name: 'day',
                    type: 'time',
                    enlarged: true
                }
            ],
            layers: [
                {
                    type: 'heatmap',
                    config: {
                        dataId: 'covid19',
                        columns: {
                            lat: 'latitude',
                            lng: 'longitude'
                        },
                        isVisible: true,
                        visConfig: {
                            radius: 100,
                        },
                    },
                }
            ]
        }
    };

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
                    config: sampleConfig
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
