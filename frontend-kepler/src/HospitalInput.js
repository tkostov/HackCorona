import React from 'react';
import TextField from '@material-ui/core/TextField';
import { makeStyles } from '@material-ui/core/styles';
import CountrySelect from './CountrySelect'
import {Link} from "react-router";
import Button from "@material-ui/core/Button";

import useSwr from "swr";

import Grid from '@material-ui/core/Grid';
import DateFnsUtils from '@date-io/date-fns';
import {
  MuiPickersUtilsProvider,
  KeyboardTimePicker,
  KeyboardDatePicker,
} from '@material-ui/pickers';

import Title from './Title';
import { useTheme } from '@material-ui/core/styles';
import { LineChart, Line, XAxis, YAxis, Label, ResponsiveContainer } from 'recharts';

const useStyles = makeStyles((theme) => ({
    root: {
        '& .MuiTextField-root': {
            margin: theme.spacing(1),
            width: '25ch',
        },
    },
}));


// Generate Sales Data
function createData(time, amount) {
  return { time, amount };
}

const data = [
  createData('2020-03-27', 1000),
  createData('2020-03-28', 1300),
  createData('2020-03-29', 1700),
  createData('2020-03-30', 2200),
  createData('2020-03-31', 2800),
  createData('2020-04-01', 3500),
  createData('2020-04-02', 4300),
  createData('2020-04-03', 5200),
  createData('2020-04-04', 0),
  createData('2020-04-05', 0),
  createData('2020-04-06', 0),
  createData('2020-04-07', 0),
];

function myFunction() {
    const data = [
  createData('2020-03-27', 1000),
  createData('2020-03-28', 1300),
  createData('2020-03-29', 1700),
  createData('2020-03-30', 2200),
  createData('2020-03-31', 2800),
  createData('2020-04-01', 3500),
  createData('2020-04-02', 4300),
  createData('2020-04-03', 5200),
  createData('2020-04-04', 6200),
  createData('2020-04-05', 7300),
  createData('2020-04-06', 8400),
  createData('2020-04-07', 9600),
];
}

export default function HospitalInput() {
  const theme = useTheme();
  const classes = useStyles();
  const [selectedDate, setSelectedDate] = React.useState(new Date('2020-04-05T00:00:00'));
  const handleDateChange = (date) => {
    setSelectedDate(date);
  };
  // const {data} = useSwr("New York", async () => {
  //       const response = await fetch(
  //           "http://ec2-3-122-224-7.eu-central-1.compute.amazonaws.com:8080/info?city=New%20York&country=US&state=New%20York&needs=1"
  //       );
  //       const data = await response.json();
  //       return data;
  //   });
    console.log(data)
  return (
    <React.Fragment>
        <form className={classes.root} noValidate autoComplete="off">
            <CountrySelect/>
            <TextField required id="standard-basic" label="Hospital" />
            <TextField required id="standard-basic" label="City " oninput="myFunction()" />
            <TextField required id="standard-basic" label="State" />
        </form>
        Items needed by:
        <form className={classes.root} noValidate autoComplete="off">
            <MuiPickersUtilsProvider utils={DateFnsUtils}>
              <KeyboardDatePicker
                  disableToolbar
                  variant="inline"
                  format="MM/dd/yyyy"
                  margin="normal"
                  id="date-picker-inline"
                  label="Date picker inline"
                  value={selectedDate}
                  onChange={handleDateChange}
                  KeyboardButtonProps={{
                    'aria-label': 'change date',
                  }}
                />
            </MuiPickersUtilsProvider>
        </form>
        <form className={classes.root} noValidate autoComplete="off">
             <TextField required id="standard-basic" label="Item" />
             <TextField required id="standard-basic" label="Quantity" />
        </form>
        <form className={classes.root} noValidate autoComplete="off">
             <TextField id="standard-basic" label="Item" />
             <TextField id="standard-basic" label="Quantity" />
        </form>
        <form className={classes.root} noValidate autoComplete="off">
             <TextField id="standard-basic" label="Item" />
             <TextField id="standard-basic" label="Quantity" />
        </form>
        <form className={useStyles.root} noValidate autoComplete="off">
          <Title>COVID-19 Cases in your Region</Title>
          <ResponsiveContainer width='60%' aspect={4.0/3.0}>
            <LineChart
              data={data}
              margin={{
                top: 16,
                right: 16,
                bottom: 0,
                left: 24,
              }}
            >
              <XAxis dataKey="time" stroke={theme.palette.text.secondary} />
              <YAxis stroke={theme.palette.text.secondary}>
                <Label
                  angle={270}
                  position="left"
                  style={{ textAnchor: 'middle', fill: theme.palette.text.primary }}
                >
                  COVID-19 Cases
                </Label>
              </YAxis>
              <Line type="monotone" dataKey="amount" stroke={theme.palette.primary.main} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </form>

         <Link to={"/"}>
             <Button
                 type="submit"
                 fullWidth
                 variant="contained"
                 color="primary"
                 className={classes.submit}
                 >
             Submit
         </Button></Link>

    </React.Fragment>
  );
}


// export default function HospitalInput() {
//     const classes = useStyles();
//     const theme = useTheme();
//
//     return (
//
//
//         <form className={classes.root} noValidate autoComplete="off">
//         <TextField required id="standard-basic" label="Hospital" />
//         <TextField required id="standard-basic" label="City" />
//         <TextField required id="standard-basic" label="State" />
//
//         <CountrySelect/>
//         <TextField required id="standard-basic" label="ICUs" />
//         <TextField required id="standard-basic" label="ICU capacity" />
//         <>
//             <label>Required immediately</label>
//         </>
//
//         <Link to={"/"}>
//             <Button
//                 type="submit"
//                 fullWidth
//                 variant="contained"
//                 color="primary"
//                 className={classes.submit}
//                 >
//             Submit
//             </Button></Link>
//         </form>
//
//   );
// }