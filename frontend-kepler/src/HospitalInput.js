import React from 'react';
import TextField from '@material-ui/core/TextField';
import { makeStyles } from '@material-ui/core/styles';
import CountrySelect from './CountrySelect'
import {Link} from "react-router";
import Button from "@material-ui/core/Button";


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

  console.log(data)
  return (
    <React.Fragment>
      <Title>Today</Title>
      <ResponsiveContainer>
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