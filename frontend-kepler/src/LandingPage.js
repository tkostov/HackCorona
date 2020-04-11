import Background from './background.jpeg';
import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid'
import Paper from '@material-ui/core/Paper';
import Button from '@material-ui/core/Button';
import Need from './need.jpg';
import Supply from './supply.jpeg';
import Map from './map.png';
import { Link } from "react-router-dom";
import Logo from './logo.png';


var sectionStyle = {
    backgroundPosition: 'center',
    backgroundSize: 'cover',
    backgroundRepeat: 'no-repeat',
    width: '100vw',
    height: '100vh',
    backgroundImage: `url(${Background})`
};

const useStyles = makeStyles((theme) => ({
    root: {
        flexGrow: 1,
        height: "100%"
    },
    control: {
        padding: theme.spacing(2),
    },
    paper1 : {
        height: 210,
        width: 300,
        backgroundImage: `url(${Need})`
    },
    paper2 : {
        height: 210,
        width: 300,
        backgroundImage: `url(${Supply})`
    },
    paper3 : {
        height: 210,
        width: 300,
        backgroundImage: `url(${Map})`,
        backgroundPosition: 'center',
        backgroundSize: 'cover',
        backgroundRepeat: 'no-repeat'
    }
}));

export default function LandingPage(){
    const classes = useStyles();

    return (
        <div className="landing-page" style={ sectionStyle }>
            <Grid container className={classes.root} spacing={2} alignItems="center" alignContent="center"
                  justify="center">
                <Grid item xs={12}>
                    <center><img src={Logo} width="600" alt=""/></center>
                </Grid>
                <Grid item xs={12}>
                    <Grid container justify="center" spacing={3}>
                        <Grid key={0} item>
                            <Paper className={classes.paper1} />
                        </Grid>
                        <Grid key={1} item>
                            <Paper className={classes.paper2} />
                        </Grid>
                        <Grid key={2} item>
                            <Paper className={classes.paper3} />
                        </Grid>
                    </Grid>
                </Grid>
                <Grid item xs={12}>
                    <Grid container justify="center" spacing={3}>
                        <Grid key={0} item>
                            <Link to={'./login'}>
                                <Button variant="contained" color="primary" style={{maxWidth: '300px', minWidth: '300px'}}>
                                    demand resources
                                </Button>
                            </Link>
                        </Grid>
                        <Grid key={1} item>
                            <Link to={'./supply'}>
                                <Button variant="contained" color="primary" style={{maxWidth: '300px', minWidth: '300px'}}>
                                    provide resources
                                </Button>
                            </Link>
                        </Grid>
                        <Grid key={2} item>
                            <Link to={'./map'}>
                                <Button variant="contained" color="primary" style={{maxWidth: '300px', minWidth: '300px'}}>
                                    see outbreak
                                </Button>
                            </Link>
                        </Grid>
                    </Grid>
                </Grid>
            </Grid>
        </div>
    )
}