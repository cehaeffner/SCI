data {
  int<lower=1> nt;
  int<lower=1> ns;
  int<lower=1, upper=ns> subid[nt];
  int<lower=1> Tsubj[ns];
  real<lower=0> gain[nt];
  real<lower=0> pgain[nt];
  real cert[nt];
  int<lower=0, upper=1> gamble[nt];
}

parameters {
  real am;
  real gm;
  real<lower=0> as;
  real<lower=0> gs;
  vector[ns] alph_raw;
  vector[ns] gamma_raw;
}

transformed parameters {
  vector<lower=0>[ns] alph  = exp(am + as * alph_raw);
  vector<lower=0>[ns] gamma = exp(gm + gs * gamma_raw);
}

model {
  am ~ normal(log(0.65), 1.0);
  gm ~ normal(0.0, 1.0);
  as ~ lognormal(0.0, 1.0);
  gs ~ lognormal(0.0, 1.0);
  alph_raw  ~ normal(0.0, 1.0);
  gamma_raw ~ normal(0.0, 1.0);

  for (t in 1:nt) {
    int sid;
    real svSafe;
    real svGamble;
    sid      = subid[t];
    svSafe   = pow(cert[t],  alph[sid]);
    svGamble = pgain[t] * pow(gain[t], alph[sid]);
    gamble[t] ~ bernoulli_logit(gamma[sid] * (svGamble - svSafe));
  }
}

generated quantities {
  vector[nt] log_lik;
  int y_pred[nt];

  for (t in 1:nt) {
    int sid;
    real svSafe;
    real svGamble;
    real p_gamble;
    sid       = subid[t];
    svSafe    = pow(cert[t],  alph[sid]);
    svGamble  = pgain[t] * pow(gain[t], alph[sid]);
    p_gamble  = gamma[sid] * (svGamble - svSafe);
    log_lik[t] = bernoulli_logit_lpmf(gamble[t] | p_gamble);
    y_pred[t]  = bernoulli_logit_rng(p_gamble);
  }
}
