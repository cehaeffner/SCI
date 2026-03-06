data {
  // Shared
  int<lower=1> ns;

  // RDM data
  int<lower=1> nt_rdm;
  int<lower=1, upper=ns> subid_rdm[nt_rdm];
  real<lower=0> gain[nt_rdm];
  real<lower=0> pgain[nt_rdm];
  real cert[nt_rdm];
  real<lower=0> alott[nt_rdm];              // ambiguity level
  int<lower=0, upper=1> gamble[nt_rdm];

  // DD data
  int<lower=1> nt_dd;
  int<lower=1, upper=ns> subid_dd[nt_dd];
  real<lower=0> amount_later[nt_dd];
  real<lower=0> amount_sooner[nt_dd];
  real<lower=0> delay_later[nt_dd];
  int<lower=0, upper=1> choice_dd[nt_dd];
}

parameters {
  // Shared gamma (stochasticity)
  real gm;
  real<lower=0> gs;
  vector[ns] gamma_raw;

  // RDM alpha (risk preference)
  real am;
  real<lower=0> as;
  vector[ns] alph_raw;

  // RDM beta (ambiguity aversion)
  real bm;
  real<lower=0> bs;
  vector[ns] beta_raw;

  // DD kappa (discounting)
  real km;
  real<lower=0> ks;
  vector[ns] kapp_raw;
}

transformed parameters {
  vector<lower=0>[ns] gamma = exp(gm + gs * gamma_raw); // shared stochasticity
  vector<lower=0>[ns] alph  = exp(am + as * alph_raw);  // risk preference
  vector[ns] beta = bm + bs * beta_raw;                 // ambiguity aversion (can be negative)
  vector<lower=0>[ns] kapp  = exp(km + ks * kapp_raw);  // discounting rate
}

model {
  // Hyperpriors
  gm ~ normal(0.0, 1.0);
  gs ~ lognormal(0.0, 1.0);
  am ~ normal(log(0.65), 1.0);
  as ~ lognormal(0.0, 1.0);
  bm ~ normal(0.65, 1.0);
  bs ~ lognormal(0.0, 1.0);
  km ~ normal(-3.0, 1.0);
  ks ~ lognormal(0.0, 1.0);

  // Raw priors
  gamma_raw ~ normal(0.0, 1.0);
  alph_raw  ~ normal(0.0, 1.0);
  beta_raw  ~ normal(0.0, 1.0);
  kapp_raw  ~ normal(0.0, 1.0);

  // RDM likelihood — softmax with ambiguity
  for (t in 1:nt_rdm) {
    int sid;
    real sv_cert;
    real sv_gamble;
    real adjustedProb;
    sid          = subid_rdm[t];
    sv_cert      = pow(cert[t], alph[sid]);
    adjustedProb = fmin(1.0, fmax(0.001, pgain[t] - beta[sid] * (alott[t] / 2.0)));
    sv_gamble    = adjustedProb * pow(gain[t], alph[sid]);
    gamble[t] ~ bernoulli_logit(gamma[sid] * (sv_gamble - sv_cert));
  }

  // DD likelihood — softmax
  for (t in 1:nt_dd) {
    int sid;
    real sv_later;
    real sv_sooner;
    sid        = subid_dd[t];
    sv_later   = pow(amount_later[t],  alph[sid]) / (1 + kapp[sid] * delay_later[t]);
    sv_sooner  = pow(amount_sooner[t], alph[sid]);
    choice_dd[t] ~ bernoulli_logit(gamma[sid] * (sv_later - sv_sooner));
  }
}

generated quantities {
  vector[nt_rdm] log_lik_rdm;
  vector[nt_dd]  log_lik_dd;
  int y_pred_rdm[nt_rdm];
  int y_pred_dd[nt_dd];

  for (t in 1:nt_rdm) {
    int sid;
    real sv_cert;
    real sv_gamble;
    real adjustedProb;
    real p_gamble;
    sid          = subid_rdm[t];
    sv_cert      = pow(cert[t], alph[sid]);
    adjustedProb = fmin(1.0, fmax(0.001, pgain[t] - beta[sid] * (alott[t] / 2.0)));
    sv_gamble    = adjustedProb * pow(gain[t], alph[sid]);
    p_gamble     = gamma[sid] * (sv_gamble - sv_cert);
    log_lik_rdm[t] = bernoulli_logit_lpmf(gamble[t] | p_gamble);
    y_pred_rdm[t]  = bernoulli_logit_rng(p_gamble);
  }

  for (t in 1:nt_dd) {
    int sid;
    real sv_later;
    real sv_sooner;
    real p_later;
    sid        = subid_dd[t];
    sv_later   = pow(amount_later[t],  alph[sid]) / (1 + kapp[sid] * delay_later[t]);
    sv_sooner  = pow(amount_sooner[t], alph[sid]);
    p_later    = gamma[sid] * (sv_later - sv_sooner);
    log_lik_dd[t] = bernoulli_logit_lpmf(choice_dd[t] | p_later);
    y_pred_dd[t]  = bernoulli_logit_rng(p_later);
  }
}
