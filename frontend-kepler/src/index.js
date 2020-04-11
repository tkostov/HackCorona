import React from "react";
import ReactDOM from "react-dom";
import { BrowserRouter, Route, Switch } from "react-router-dom";
import HospitalInput from "./HospitalInput";
import NeedMap from "./NeedMap";


// pages for this product
import LandingPage from "./LandingPage";
import Map from "./Map";
import SignIn from "./SignIn";

ReactDOM.render(
  <BrowserRouter>
    <Switch>
      <Route exact path="/" component={LandingPage} />
      <Route path="/login" component={SignIn} />
      <Route path="/map" component={Map} />
      <Route path="/input" component={HospitalInput} />
      <Route path="/supply" component={NeedMap}/>
    </Switch>
  </BrowserRouter>,
  document.getElementById("root")
);
