// Joint model: DD (logit/SVD) + RDM (logit/SVD)
// Gaussian copula on gamma_raw (DD) and gamma_raw (RDM)
// rho > 0 means consistently more sensitive/deterministic across tasks

data {
  // --- RDM ---
  int<lower=1> nt_rdm;
  int<lower=1> ns;
  int<lower=1, upper=ns> subid_rdm[nt_rdm];
  real<lower=0> gain[nt_rdm];
  real<lower=0> pgain[nt_rdm];
  real cert[nt_rdm];
  real<lower=0> alott_rdm[nt_rdm];
  int<lower=0, upper=1> gamble[nt_rdm];

  // --- DD ---
  int<lower=1> nt_dd;
  int<lower=1, upper=ns> subid_dd[nt_dd];
  real<lower=0> delay_later[nt_dd];
  real<lower=0> amount_later[nt_dd];
  real<lower=0> amount_sooner[nt_dd];
  int<lower=0, upper=1> choice[nt_dd];
  real<lower=0> alph_dd[ns];
}

parameters {
  // --- RDM-specific ---
  real am;
  real bm;
  real<lower=0> as;
  real<lower=0> bs;
  vector[ns] alph_raw;
  vector[ns] beta_raw;

  // --- DD-specific ---
  real km;
  real<lower=0> ks;
  vector[ns] kapp_raw;

  // --- Shared gamma (copula) ---
  real gm_rdm;
  real gm_dd;
  real<lower=0> gs_rdm;
  real<lower=0> gs_dd;
  vector[ns] gamma_raw_rdm;
  vector[ns] gamma_raw_dd;
  real<lower=-1, upper=1> rho;
}

transformed parameters {
  vector<lower=0>[ns] alph     = exp(am + as * alph_raw);
  vector[ns]          beta     = bm + bs * beta_raw;
  vector<lower=0>[ns] kapp     = exp(km + ks * kapp_raw);
  vector<lower=0>[ns] gamma_rdm = exp(gm_rdm + gs_rdm * gamma_raw_rdm);
  vector<lower=0>[ns] gamma_dd  = exp(gm_dd  + gs_dd  * gamma_raw_dd);
}

model {
  // --- RDM priors ---
  am ~ normal(log(0.65), 1.0);
  bm ~ normal(0.65, 1.0);
  as ~ lognormal(0.0, 1.0);
  bs ~ lognormal(0.0, 1.0);
  alph_raw ~ normal(0.0, 1.0);
  beta_raw  ~ normal(0.0, 1.0);

  // --- DD priors ---
  km ~ normal(-3.0, 1.0);
  ks ~ lognormal(0.0, 1.0);
  kapp_raw ~ normal(0.0, 1.0);

  // --- Gamma priors ---
  gm_rdm ~ normal(0.0, 1.0);
  gm_dd  ~ normal(0.0, 1.0);
  gs_rdm ~ lognormal(0.0, 1.0);
  gs_dd  ~ lognormal(0.0, 1.0);

  // --- Gaussian copula on gamma_raw_rdm and gamma_raw_dd ---
  rho ~ uniform(-1, 1);
  for (s in 1:ns) {
    real u = gamma_raw_rdm[s];
    real v = gamma_raw_dd[s];
    target += -0.5 * log1m(square(rho))
              - (square(u) - 2 * rho * u * v + square(v))
                / (2 * (1 - square(rho)));
  }

  // --- RDM likelihood ---
  for (t in 1:nt_rdm) {
    int sid = subid_rdm[t];
    real adjustedProb = fmin(1.0, fmax(0.001, pgain[t] - beta[sid] * (alott_rdm[t] / 2.0)));
    real svSafe   = pow(cert[t], alph[sid]);
    real svGamble = adjustedProb * pow(gain[t], alph[sid]);
    gamble[t] ~ bernoulli_logit(gamma_rdm[sid] * (svGamble - svSafe));
  }

  // --- DD likelihood ---
  for (t in 1:nt_dd) {
    int sid = subid_dd[t];
    real sv_later  = pow(amount_later[t],  alph_dd[sid]) / (1 + kapp[sid] * delay_later[t]);
    real sv_sooner = pow(amount_sooner[t], alph_dd[sid]);
    choice[t] ~ bernoulli_logit(gamma_dd[sid] * (sv_later - sv_sooner));
  }
}

generated quantities {
  real rho_out = rho;

  // RDM — matches RDM-SVD-H style
  vector[nt_rdm] log_lik_rdm;
  int y_pred_rdm[nt_rdm];

  // DD — matches DD-SVD-H style
  vector[nt_dd] log_lik_dd;
  int y_pred_dd[nt_dd];

  for (t in 1:nt_rdm) {
    int sid = subid_rdm[t];
    real adjustedProb = fmin(1.0, fmax(0.001, pgain[t] - beta[sid] * (alott_rdm[t] / 2.0)));
    real svSafe   = pow(cert[t], alph[sid]);
    real svGamble = adjustedProb * pow(gain[t], alph[sid]);
    real p_gamble = gamma_rdm[sid] * (svGamble - svSafe);
    log_lik_rdm[t] = bernoulli_logit_lpmf(gamble[t] | p_gamble);
    y_pred_rdm[t]  = bernoulli_logit_rng(p_gamble);
  }

  for (t in 1:nt_dd) {
    int sid = subid_dd[t];
    real sv_later  = pow(amount_later[t],  alph_dd[sid]) / (1 + kapp[sid] * delay_later[t]);
    real sv_sooner = pow(amount_sooner[t], alph_dd[sid]);
    real p_later   = gamma_dd[sid] * (sv_later - sv_sooner);
    log_lik_dd[t] = bernoulli_logit_lpmf(choice[t] | p_later);
    y_pred_dd[t]  = bernoulli_logit_rng(p_later);
  }
}
