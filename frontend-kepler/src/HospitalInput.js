import React from 'react';
import TextField from '@material-ui/core/TextField';
import {makeStyles} from '@material-ui/core/styles';
import CountrySelect from './CountrySelect'
import Button from "@material-ui/core/Button";
import {Link} from 'react-router-dom';
import useSwr from "swr";

import Grid from '@material-ui/core/Grid';
import DateFnsUtils from '@date-io/date-fns';
import {
    MuiPickersUtilsProvider,
    KeyboardTimePicker,
    KeyboardDatePicker,
} from '@material-ui/pickers';

import Title from './Title';
import {useTheme} from '@material-ui/core/styles';
import {LineChart, Line, XAxis, YAxis, Label, ResponsiveContainer} from 'recharts';
import Container from "@material-ui/core/Container";

const useStyles = makeStyles((theme) => ({
    root: {
        '& .MuiTextField-root': {
            margin: theme.spacing(1),
            width: '25ch',
        },
    },
}));

export default class HospitalInput extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            selectedDate: "2020-04-05T00:00:00",
            city: "",
            state: "",
            zip: "",
            country: "",
            data: []
        };
        this.updateData = this.updateData.bind(this);
        this.updateDay = this.updateDay.bind(this);
        this.updateCity = this.updateCity.bind(this);
        this.updateZip = this.updateZip.bind(this);
        this.updateState = this.updateState.bind(this);
    }

    updateDay(date) {
        console.log(date);
        this.state.selectedDate = date;
    }

    updateCity(city) {
        this.state.city = city;
        this.updateData();
    }

    updateZip(zip) {
        this.state.zip = zip;
        this.updateData();
    }

    updateState(state) {
        this.state.state = state;
        this.updateData();
    }

    updateData() {
        fetch(`http://localhost:8080/info?city=${this.state.city}&country=US&state=${this.state.state}&needs=1`)
            .then(response => response.json())
            .then(data => this.setState({
                selectedDate: this.state.selectedDate,
                data: [...data]
            }));
    }

    render() {
        return (
            <React.Fragment>
                <Grid container spacing={0} alignItems="center" justify="center">
                    <Grid item center xs={5}>
                        <Container component="main" maxWidth="xs">
                            <form noValidate autoComplete="off">
                                <TextField required id="standard-basic" label="Hospital"/>
                                <TextField required id="standard-basic" label="City "
                                           onChange={(e) => this.updateCity(e.target.value)}/>
                                <TextField required id="standard-basic" label="Postcode "
                                           onChange={(e) => this.updateZip(e.target.value)}/>
                                <TextField required id="standard-basic" label="State"
                                           onChange={(e) => this.updateState(e.target.value)}/>
                                <CountrySelect/>
                                <MuiPickersUtilsProvider utils={DateFnsUtils}>
                                    <KeyboardDatePicker disableToolbar variant="inline" format="dd.MM.yyyy"
                                                        margin="normal"
                                                        id="date-picker-inline" label="Items needed by:"
                                                        value={this.state.selectedDate} onChange={this.updateDay}
                                                        KeyboardButtonProps={
                                                            {
                                                                'aria-label':
                                                                    'change date',
                                                            }
                                                        }
                                    />
                                </MuiPickersUtilsProvider>
                            </form>
                            <form
                                noValidate
                                autoComplete="off">
                                <TextField required id="standard-basic" label="Item"/>
                                <TextField required id="standard-basic" label="Quantity"/>
                            </form>
                            <form
                                noValidate
                                autoComplete="off">
                                <TextField
                                    id="standard-basic"
                                    label="Item"/>
                                <TextField
                                    id="standard-basic"
                                    label="Quantity"/>
                            </form>
                            <form
                                noValidate
                                autoComplete="off">
                                <TextField
                                    id="standard-basic"
                                    label="Item"/>
                                <TextField
                                    id="standard-basic"
                                    label="Quantity"/>
                            </form>
                            <Link
                                to={"/map"}>
                                <Button
                                    type="submit"
                                    fullWidth
                                    variant="contained"
                                    color="primary">
                                    Submit
                                </Button></
                                Link>
                        </Container>
                    </Grid>
                    <Grid
                        item
                        center
                        xs={7}>{
                        this.state.data.length > 0 &&
                        <form
                            className={useStyles.root}
                            noValidate
                            autoComplete="off">
                            <Title> COVID - 19
                                Cases in your
                                Region </Title>
                            <ResponsiveContainer
                                width='60%'
                                aspect={4.0 / 3.0
                                }>
                                <LineChart
                                    data={this.state.data}
                                    margin={
                                        {
                                            top: 16,
                                            right:
                                                16,
                                            bottom:
                                                0,
                                            left:
                                                24,
                                        }
                                    }
                                >
                                    <XAxis
                                        dataKey="date"
                                    />
                                    <YAxis>
                                        <Label
                                            angle={270}
                                            position="left"
                                            style={
                                                {
                                                    textAnchor: 'middle'
                                                }
                                            }
                                        >
                                        </Label>
                                    </YAxis>
                                    <Line
                                        type="monotone"
                                        dataKey="cases"
                                        dot={false}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </form>
                    }
                    </ Grid>

                </Grid>
            </React.Fragment>
        );
    }
}
