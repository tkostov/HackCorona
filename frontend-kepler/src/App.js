import React from "react";
import { Link } from 'react-router-dom';


export default class App extends React.Component {
    render() {
        return (
            <div>
            <ul>
            <li><Link to={"/"}>Home</Link></li>
            <li><Link to={"/map"}>Map</Link></li>
            <li><Link to={"/login"}>Hospital Input</Link></li>
            </ul>
    </div>
    )
    }
}