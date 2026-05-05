// Joint model: DD (softmax/SVR) + RDM (softmax/SVR)
// Gaussian copula on mu_raw (DD) and mu_raw (RDM)
// Both mu are noise params: rho > 0 means consistently noisier across tasks

functions {
  real utility_log_trials(int[] obs, int[] subid, real[] log_pr, int nt) {
    real llhC = 0.0;
    for (t in 1:nt) {
      real log_p;
      if (log_pr[t] == log(1.0)) {
        log_p = log1m(machine_precision());
      } else if (log_pr[t] == log(0.0)) {
        log_p = log(machine_precision());
      } else {
        log_p = log_pr[t];
      }
      llhC += obs[t] * log_p + (1 - obs[t]) * log1m_exp(log_p);
    }
    return llhC;
  }
}

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

  // --- Shared mu (copula) ---
  real mm_rdm;
  real mm_dd;
  real<lower=0> ms_rdm;
  real<lower=0> ms_dd;
  vector[ns] mu_raw_rdm;
  vector[ns] mu_raw_dd;
  real<lower=-1, upper=1> rho;
}

transformed parameters {
  vector<lower=0>[ns] alph   = exp(am + as * alph_raw);
  vector[ns]          beta   = bm + bs * beta_raw;
  vector<lower=0>[ns] kapp   = exp(km + ks * kapp_raw);
  vector<lower=0>[ns] mu_rdm = exp(mm_rdm + ms_rdm * mu_raw_rdm);
  vector<lower=0>[ns] mu_dd  = exp(mm_dd  + ms_dd  * mu_raw_dd);
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

  // --- Mu priors ---
  mm_rdm ~ normal(0.0, 1.0);
  mm_dd  ~ normal(0.0, 1.0);
  ms_rdm ~ lognormal(0.0, 1.0);
  ms_dd  ~ lognormal(0.0, 1.0);

  // --- Gaussian copula on mu_raw_rdm and mu_raw_dd ---
  rho ~ uniform(-1, 1);
  for (s in 1:ns) {
    real u = mu_raw_rdm[s];
    real v = mu_raw_dd[s];
    target += -0.5 * log1m(square(rho))
              - (square(u) - 2 * rho * u * v + square(v))
                / (2 * (1 - square(rho)));
  }

  // --- RDM likelihood ---
  {
    real log_pr[nt_rdm];
    for (t in 1:nt_rdm) {
      int sid = subid_rdm[t];
      real adjustedProb = fmin(1.0, fmax(0.001, pgain[t] - beta[sid] * (alott_rdm[t] / 2.0)));
      real log_svSafe   = alph[sid] * log(cert[t]);
      real log_svGamble = log(adjustedProb) + alph[sid] * log(gain[t]);
      log_pr[t] = (log_svGamble / mu_rdm[sid])
                  - log_sum_exp(log_svGamble / mu_rdm[sid], log_svSafe / mu_rdm[sid]);
    }
    target += utility_log_trials(gamble, subid_rdm, log_pr, nt_rdm);
  }

  // --- DD likelihood ---
  {
    real log_pr[nt_dd];
    for (t in 1:nt_dd) {
      int sid = subid_dd[t];
      real log_sv_later  = alph_dd[sid] * log(amount_later[t])
                           - log(1 + kapp[sid] * delay_later[t]);
      real log_sv_sooner = alph_dd[sid] * log(amount_sooner[t]);
      log_pr[t] = (log_sv_later / mu_dd[sid])
                  - log_sum_exp(log_sv_later / mu_dd[sid], log_sv_sooner / mu_dd[sid]);
    }
    target += utility_log_trials(choice, subid_dd, log_pr, nt_dd);
  }
}

generated quantities {
  real rho_out = rho;

  // RDM — matches RDM-SVR style
  vector[nt_rdm] log_lik_rdm;
  int y_pred_rdm[nt_rdm];

  // DD — matches DD-SVR-H style
  vector[nt_dd] log_lik_dd;
  int y_pred_dd[nt_dd];

  for (t in 1:nt_rdm) {
    int sid = subid_rdm[t];
    real adjustedProb = fmin(1.0, fmax(0.001, pgain[t] - beta[sid] * (alott_rdm[t] / 2.0)));
    real log_svSafe   = alph[sid] * log(cert[t]);
    real log_svGamble = log(adjustedProb) + alph[sid] * log(gain[t]);
    real log_p = (log_svGamble / mu_rdm[sid])
                 - log_sum_exp(log_svGamble / mu_rdm[sid], log_svSafe / mu_rdm[sid]);
    if (log_p >= log(1.0)) log_p = log1m(machine_precision());
    if (log_p <= log(0.0)) log_p = log(machine_precision());
    log_lik_rdm[t] = gamble[t] * log_p + (1 - gamble[t]) * log1m_exp(log_p);
    y_pred_rdm[t]  = bernoulli_rng(exp(log_p));
  }

  for (t in 1:nt_dd) {
    int sid = subid_dd[t];
    real log_sv_later  = alph_dd[sid] * log(amount_later[t])
                         - log(1 + kapp[sid] * delay_later[t]);
    real log_sv_sooner = alph_dd[sid] * log(amount_sooner[t]);
    real log_p = (log_sv_later / mu_dd[sid])
                 - log_sum_exp(log_sv_later / mu_dd[sid], log_sv_sooner / mu_dd[sid]);
    if (log_p >= log(1.0)) log_p = log1m(machine_precision());
    if (log_p <= log(0.0)) log_p = log(machine_precision());
    log_lik_dd[t] = choice[t] * log_p + (1 - choice[t]) * log1m_exp(log_p);
    y_pred_dd[t]  = bernoulli_rng(exp(log_p));
  }
}
