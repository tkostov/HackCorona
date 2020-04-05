import React from 'react';
import TextField from '@material-ui/core/TextField';
import { makeStyles } from '@material-ui/core/styles';
import CountrySelect from './CountrySelect'
import {Link} from "react-router";
import Button from "@material-ui/core/Button";


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
  createData('00:00', 0),
  createData('03:00', 300),
  createData('06:00', 600),
  createData('09:00', 800),
  createData('12:00', 1500),
  createData('15:00', 2000),
  createData('18:00', 2400),
  createData('21:00', 2400),
  createData('24:00', undefined),
];

export default function HospitalInput() {
  const theme = useTheme();
  const classes = useStyles();
  const [selectedDate, setSelectedDate] = React.useState(new Date('2020-04-05T00:00:00'));
  const handleDateChange = (date) => {
    setSelectedDate(date);
  };

  return (
    <React.Fragment>
        <form className={classes.root} noValidate autoComplete="off">
            <CountrySelect/>
            <TextField required id="standard-basic" label="Hospital" />
            <TextField required id="standard-basic" label="City" />
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
          <Title>Today</Title>
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
                  Sales ($)
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