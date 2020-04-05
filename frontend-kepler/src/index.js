import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import HospitalInput from './HospitalInput'
import SupplierInput from './SupplierInput'
import Map from './Map';
import SignIn from './SignIn';
import * as serviceWorker from './serviceWorker';
import { BrowserRouter, Route, Switch } from 'react-router-dom';

ReactDOM.render(
    <BrowserRouter>
    <Switch>
        <Route exact path="/" component={App}/>
        <Route path="/map" component={Map}/>
        <Route path="/login" component={SignIn}/>
        <Route path="/input" component={HospitalInput}/>
        <Route path="/supplier" component={SupplierInput}/>
    </Switch>
    </BrowserRouter>
    , document.getElementById('root'));

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
