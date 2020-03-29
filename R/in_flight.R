#TODO data to be put by the python script
horizon <- 13;
fit_horizon <- 15;
is_fit <- 1; # Get from 

region_id = 123;
library(pomp);
library(tidyverse);
library(foreach);
library(doParallel);
library(iterators);
library(doRNG);
registerDoParallel();
registerDoRNG(887851050L);


populations <- c(region=10628972);


read_csv("https://kingaa.github.io/sbied/ebola/ebola_data.csv") -> dat

dat

set.seed(594709947L)
library(tidyverse)
library(pomp)
stopifnot(packageVersion("pomp")>="2.1")
options(
  keep.source=TRUE,
  stringsAsFactors=FALSE,
  encoding="UTF-8"
)

## ----rproc,include=FALSE-------------------------------------------------
rSim <- Csnippet("
  double lambda, beta;
  double *E = &E1;
  beta = R0 * gamma; // Transmission rate
  lambda = beta * I / N; // Force of infection
  int i;

  // Transitions
  // From class S
  double transS = rbinom(S, 1.0 - exp(- lambda * dt)); // No of infections
  // From class E
  double transE[nstageE]; // No of transitions between classes E
  for(i = 0; i < nstageE; i++){
    transE[i] = rbinom(E[i], 1.0 - exp(-nstageE * alpha * dt));
  }
  // From class I
  double transI = rbinom(I, 1.0 - exp(-gamma * dt)); // No of transitions I->R

  // Balance the equations
  S -= transS;
  E[0] += transS - transE[0];
  for(i=1; i < nstageE; i++) {
    E[i] += transE[i-1] - transE[i];
  }
  I += transE[nstageE-1] - transI;
  R += transI;
  N_EI += transE[nstageE-1]; // No of transitions from E to I
  N_IR += transI; // No of transitions from I to R
")

rInit <- Csnippet("
  double m = N/(S_0+E_0+I_0+R_0);
  double *E = &E1;
  int j;
  S = nearbyint(m*S_0);
  for (j = 0; j < nstageE; j++) E[j] = nearbyint(m*E_0/nstageE);
  I = nearbyint(m*I_0);
  R = nearbyint(m*R_0);
  N_EI = 0;
  N_IR = 0;
")

## ----skel,include=FALSE--------------------------------------------------
skel <- Csnippet("
  double lambda, beta;
  const double *E = &E1;
  double *DE = &DE1;
  beta = R0 * gamma; // Transmission rate
  lambda = beta * I / N; // Force of infection
  int i;

  // Balance the equations
  DS = - lambda * S;
  DE[0] = lambda * S - nstageE * alpha * E[0];
  for (i=1; i < nstageE; i++)
    DE[i] = nstageE * alpha * (E[i-1]-E[i]);
  DI = nstageE * alpha * E[nstageE-1] - gamma * I;
  DR = gamma * I;
  DN_EI = nstageE * alpha * E[nstageE-1];
  DN_IR = gamma * I;
")

## ----measmodel,include=FALSE---------------------------------------------
dObs <- Csnippet("
  double f;
  if (k > 0.0)
    f = dnbinom_mu(nearbyint(cases),1.0/k,rho*N_EI,1);
  else
    f = dpois(nearbyint(cases),rho*N_EI,1);
  lik = (give_log) ? f : exp(f);
")

rObs <- Csnippet("
  if (k > 0) {
    cases = rnbinom_mu(1.0/k,rho*N_EI);
  } else {
    cases = rpois(rho*N_EI);
  }")

## ----pomp-construction,include=FALSE-------------------------------------
covidModel <- function (country=c("region"),
                        timestep = 0.1, nstageE = 3) {
  
  ctry <- match.arg(country)
  pop <- unname(populations[ctry])
  nstageE <- as.integer(nstageE)
  
  globs <- paste0("static int nstageE = ",nstageE,";")
  
  dat <- subset(dat,country==ctry,select=-country)
  
  ## Create the pomp object
  dat %>%
    select(week,cases) %>%
    pomp(
      times="week",
      t0=min(dat$week)-1,
      globals=globs,
      accumvars=c("N_EI","N_IR"),
      statenames=c("S",sprintf("E%1d",seq_len(nstageE)),
                   "I","R","N_EI","N_IR"),
      paramnames=c("N","R0","alpha","gamma","rho","k",
                   "S_0","E_0","I_0","R_0"),
      dmeasure=dObs, rmeasure=rObs,
      rprocess=discrete_time(step.fun=rSim, delta.t=timestep),
      skeleton=vectorfield(skel),
      partrans=parameter_trans(
        log=c("R0","k"),logit="rho",
        barycentric=c("S_0","E_0","I_0","R_0")),
      rinit=rInit
    ) -> po
}

covidModel("region") -> region_model

## ----load-profile,echo=FALSE---------------------------------------------
# options(stringsAsFactors=FALSE)
# read_csv("https://kingaa.github.io/sbied/ebola/ebola-profiles.csv") -> profs


## ----forecasts1----------------------------------------------------------

options(stringsAsFactors=FALSE)
set.seed(988077383L)

## forecast horizon


## Weighted quantile function
wquant <- function (x, weights, probs = c(0.025,0.5,0.975)) {
  idx <- order(x)
  x <- x[idx]
  weights <- weights[idx]
  w <- cumsum(weights)/sum(weights)
  rval <- approx(w,x,probs,rule=1)
  rval$y
}

profs %>%
  filter(country=="SierraLeone") %>%
  select(-country,-profile,-loglik.se) %>%
  filter(loglik>max(loglik)-0.5*qchisq(df=1,p=0.99)) %>%
  gather(parameter,value) %>%
  group_by(parameter) %>%
  summarize(min=min(value),max=max(value)) %>%
  ungroup() %>%
  filter(parameter!="loglik") %>%
  column_to_rownames("parameter") %>%
  as.matrix() -> ranges

sobolDesign(lower=ranges[,'min'],
            upper=ranges[,'max'],
            nseq=20) -> params
plot(params)



## ----forecasts2----------------------------------------------------------


foreach(p=iter(params,by='row'),
        .inorder=FALSE,
        .combine=bind_rows
) %dopar% {
  
  library(pomp)
  
  M1 <- ebolaModel("SierraLeone")
  
  M1 %>% pfilter(params=p,Np=2000,save.states=TRUE) -> pf
  
  pf@saved.states %>%               # latent state for each particle
    tail(1) %>%                     # last timepoint only
    melt() %>%                      # reshape and rename the state variables
    spread(variable,value) %>%
    group_by(rep) %>%
    summarize(
      S_0=S,
      E_0=E1+E2+E3,
      I_0=I,
      R_0=R
    ) %>%
    gather(variable,value,-rep) %>%
    spread(rep,value) %>%
    column_to_rownames("variable") %>%
    as.matrix() -> x
  ## the final states are now stored in 'x' as initial conditions
  
  ## set up a matrix of parameters
  pp <- parmat(unlist(p),ncol(x))
  
  ## generate simulations over the interval for which we have data
  M1 %>%
    simulate(params=pp,format="data.frame") %>%
    select(.id,week,cases) %>%
    mutate(
      period="calibration",
      loglik=logLik(pf)
    ) -> calib
  
  ## make a new 'pomp' object for the forecast simulations
  M2 <- M1
  time(M2) <- max(time(M1))+seq_len(horizon)
  timezero(M2) <- max(time(M1))
  
  ## set the initial conditions to the final states computed above
  pp[rownames(x),] <- x
  
  ## perform forecast simulations
  M2 %>%
    simulate(params=pp,format="data.frame") %>%
    select(.id,week,cases) %>%
    mutate(
      period="projection",
      loglik=logLik(pf)
    ) -> proj
  
  bind_rows(calib,proj)
} %>%
  mutate(weight=exp(loglik-mean(loglik))) %>%
  arrange(week,.id) -> sims

## look at effective sample size
ess <- with(subset(sims,week==max(week)),weight/sum(weight))
ess <- 1/sum(ess^2); ess

## compute quantiles of the forecast incidence
sims %>%
  group_by(week,period) %>%
  summarize(
    lower=wquant(cases,weights=weight,probs=0.025),
    median=wquant(cases,weights=weight,probs=0.5),
    upper=wquant(cases,weights=weight,probs=0.975)
  ) %>%
  ungroup() %>%
  mutate(date=min(dat$date)+7*(week-1)) -> simq

## ----forecast-plots,echo=FALSE-------------------------------------------
simq %>%
  ggplot(aes(x=date))+
  geom_ribbon(aes(ymin=lower,ymax=upper,fill=period),alpha=0.3,color=NA)+
  geom_line(aes(y=median,color=period))+
  geom_point(data=subset(dat,country=="SierraLeone"),
             mapping=aes(x=date,y=cases),color='black')+
  labs(y="cases")