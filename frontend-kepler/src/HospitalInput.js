import React from 'react';
import TextField from '@material-ui/core/TextField';
import { makeStyles } from '@material-ui/core/styles';
import CountrySelect from './CountrySelect'
import {Link} from "react-router";
import Button from "@material-ui/core/Button";

const useStyles = makeStyles((theme) => ({
    root: {
        '& .MuiTextField-root': {
            margin: theme.spacing(1),
            width: '25ch',
        },
    },
}));

export default function HospitalInput() {
    const classes = useStyles();

    return (
        <form className={classes.root} noValidate autoComplete="off">
        <TextField required id="standard-basic" label="Hospital" />
        <TextField required id="standard-basic" label="City" />
        <CountrySelect/>
        <TextField required id="standard-basic" label="ICUs in use" />
        <TextField required id="standard-basic" label="ICU capacity" />
        <Link to={"/"}><Button
    type="submit"
    fullWidth
    variant="contained"
    color="primary"
    className={classes.submit}
        >
        Submit
    </Button></Link>
        </form>
);
}