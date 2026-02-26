data {
  int<lower=1> nt;
  int<lower=1> ns;
  int<lower=1, upper=ns> subid[nt];
  int<lower=1> Tsubj[ns];
  real<lower=0> delay_later[nt];
  real<lower=0> amount_later[nt];
  real<lower=0> amount_sooner[nt];
  int<lower=0, upper=1> choice[nt];
  real<lower=0> alph[ns];
}

parameters {
  real km;
  real gm;
  real<lower=0> ks;
  real<lower=0> gs;
  vector[ns] kapp_raw;
  vector[ns] gamma_raw;
}

transformed parameters {
  vector<lower=0>[ns] kapp  = exp(km + ks * kapp_raw);
  vector<lower=0>[ns] gamma = exp(gm + gs * gamma_raw);
}

model {
  km ~ normal(-3.0, 1.0);
  ks ~ lognormal(0.0, 1.0);
  gm ~ normal(0.0, 1.0);
  gs ~ lognormal(0.0, 1.0);
  kapp_raw  ~ normal(0.0, 1.0);
  gamma_raw ~ normal(0.0, 1.0);

  for (t in 1:nt) {
    int sid;
    real sv_later;
    real sv_sooner;
    sid       = subid[t];
    sv_later  = pow(amount_later[t],  alph[sid]) / (1 + kapp[sid] * delay_later[t]);
    sv_sooner = pow(amount_sooner[t], alph[sid]);
    choice[t] ~ bernoulli_logit(gamma[sid] * (sv_later - sv_sooner));
  }
}

generated quantities {
  vector[nt] log_lik;
  int y_pred[nt];

  for (t in 1:nt) {
    int sid;
    real sv_later;
    real sv_sooner;
    real p_later;
    sid        = subid[t];
    sv_later   = pow(amount_later[t],  alph[sid]) / (1 + kapp[sid] * delay_later[t]);
    sv_sooner  = pow(amount_sooner[t], alph[sid]);
    p_later    = gamma[sid] * (sv_later - sv_sooner);
    log_lik[t] = bernoulli_logit_lpmf(choice[t] | p_later);
    y_pred[t]  = bernoulli_logit_rng(p_later);
  }
}
