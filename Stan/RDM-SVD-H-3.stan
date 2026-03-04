data {
  int<lower=1> nt;
  int<lower=1> ns;
  int<lower=1, upper=ns> subid[nt];
  int<lower=1> Tsubj[ns];
  real<lower=0> gain[nt];
  real<lower=0> pgain[nt];
  real cert[nt];
  real<lower=0> alott[nt];           // ambiguity level for each trial
  int<lower=0, upper=1> gamble[nt];
}

parameters {
  // Hyperparameters
  real am;          // alpha mu (risk preference)
  real gm;          // gamma mu (stochasticity)
  real bm;          // beta mu (ambiguity aversion)
  real<lower=0> as;
  real<lower=0> gs;
  real<lower=0> bs;

  // Raw parameters (non-centered)
  vector[ns] alph_raw;
  vector[ns] gamma_raw;
  vector[ns] beta_raw;
}

transformed parameters {
  vector<lower=0>[ns] alph  = exp(am + as * alph_raw);   // risk preference
  vector<lower=0>[ns] gamma = exp(gm + gs * gamma_raw);  // stochasticity
  vector[ns] beta = bm + bs * beta_raw;                  // ambiguity aversion (can be negative)
}

model {
  // Hyperpriors
  am ~ normal(log(0.65), 1.0);
  gm ~ normal(0.0, 1.0);
  bm ~ normal(0.65, 1.0);
  as ~ lognormal(0.0, 1.0);
  gs ~ lognormal(0.0, 1.0);
  bs ~ lognormal(0.0, 1.0);

  // Raw priors
  alph_raw  ~ normal(0.0, 1.0);
  gamma_raw ~ normal(0.0, 1.0);
  beta_raw  ~ normal(0.0, 1.0);

  // Local variables
  real svSafe;
  real svGamble;
  real adjustedProb;

  for (t in 1:nt) {
    int sid = subid[t];
    svSafe       = pow(cert[t], alph[sid]);
    adjustedProb = fmin(1.0, fmax(0.001, pgain[t] - beta[sid] * (alott[t] / 2.0)));
    svGamble     = adjustedProb * pow(gain[t], alph[sid]);
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
    real adjustedProb;
    real p_gamble;
    sid          = subid[t];
    svSafe       = pow(cert[t], alph[sid]);
    adjustedProb = fmin(1.0, fmax(0.001, pgain[t] - beta[sid] * (alott[t] / 2.0)));
    svGamble     = adjustedProb * pow(gain[t], alph[sid]);
    p_gamble     = gamma[sid] * (svGamble - svSafe);
    log_lik[t]   = bernoulli_logit_lpmf(gamble[t] | p_gamble);
    y_pred[t]    = bernoulli_logit_rng(p_gamble);
  }
}
