functions {
  real utility_log_trials(int[] choice, int[] subid, real[] log_pr, int nt) {
    real llhC;
    llhC = 0.0;
    for (t in 1:nt) {
      real log_p;
      if (log_pr[t] == log(1.0)) {
        log_p = log1m(machine_precision());
      } else if (log_pr[t] == log(0.0)) {
        log_p = log(machine_precision());
      } else {
        log_p = log_pr[t];
      }
      llhC = llhC + choice[t] * log_p + (1 - choice[t]) * log1m_exp(log_p);
    }
    return llhC;
  }
}

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
  // Shared mu (stochasticity)
  real mm;
  real<lower=0> ms;
  vector[ns] mu_raw;

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
  vector<lower=0>[ns] mu   = exp(mm + ms * mu_raw);    // shared stochasticity
  vector<lower=0>[ns] alph = exp(am + as * alph_raw);  // risk preference
  vector[ns] beta = bm + bs * beta_raw;                // ambiguity aversion (can be negative)
  vector<lower=0>[ns] kapp = exp(km + ks * kapp_raw);  // discounting rate
}

model {
  real log_pr_rdm[nt_rdm];
  real log_pr_dd[nt_dd];

  // Hyperpriors
  mm ~ normal(0.0, 1.0);
  ms ~ lognormal(0.0, 1.0);
  am ~ normal(log(0.65), 1.0);
  as ~ lognormal(0.0, 1.0);
  bm ~ normal(0.65, 1.0);
  bs ~ lognormal(0.0, 1.0);
  km ~ normal(-3.0, 1.0);
  ks ~ lognormal(0.0, 1.0);

  // Raw priors
  mu_raw   ~ normal(0.0, 1.0);
  alph_raw ~ normal(0.0, 1.0);
  beta_raw ~ normal(0.0, 1.0);
  kapp_raw ~ normal(0.0, 1.0);

  // RDM likelihood — Luce with ambiguity
  for (t in 1:nt_rdm) {
    int sid;
    real adjustedProb;
    real log_svSafe;
    real log_svGamble;
    sid          = subid_rdm[t];
    adjustedProb = fmin(1.0, fmax(0.001, pgain[t] - beta[sid] * (alott[t] / 2.0)));
    log_svSafe   = alph[sid] * log(cert[t]);
    log_svGamble = log(adjustedProb) + alph[sid] * log(gain[t]);
    log_pr_rdm[t] = (log_svGamble / mu[sid]) - log_sum_exp(log_svGamble / mu[sid], log_svSafe / mu[sid]);
  }
  target += utility_log_trials(gamble, subid_rdm, log_pr_rdm, nt_rdm);

  // DD likelihood — Luce
  for (t in 1:nt_dd) {
    int sid;
    real log_sv_later;
    real log_sv_sooner;
    sid           = subid_dd[t];
    log_sv_later  = alph[sid] * log(amount_later[t])  - log(1 + kapp[sid] * delay_later[t]);
    log_sv_sooner = alph[sid] * log(amount_sooner[t]);
    log_pr_dd[t]  = (log_sv_later / mu[sid]) - log_sum_exp(log_sv_later / mu[sid], log_sv_sooner / mu[sid]);
  }
  target += utility_log_trials(choice_dd, subid_dd, log_pr_dd, nt_dd);
}

generated quantities {
  vector[nt_rdm] log_lik_rdm;
  vector[nt_dd]  log_lik_dd;
  int y_pred_rdm[nt_rdm];
  int y_pred_dd[nt_dd];

  for (t in 1:nt_rdm) {
    int sid;
    real adjustedProb;
    real log_svSafe;
    real log_svGamble;
    real log_p;
    real p_gamble;
    sid          = subid_rdm[t];
    adjustedProb = fmin(1.0, fmax(0.001, pgain[t] - beta[sid] * (alott[t] / 2.0)));
    log_svSafe   = alph[sid] * log(cert[t]);
    log_svGamble = log(adjustedProb) + alph[sid] * log(gain[t]);
    log_p        = (log_svGamble / mu[sid]) - log_sum_exp(log_svGamble / mu[sid], log_svSafe / mu[sid]);
    if (log_p >= log(1.0)) log_p = log1m(machine_precision());
    else if (log_p <= log(0.0)) log_p = log(machine_precision());
    p_gamble         = exp(log_p);
    log_lik_rdm[t]   = gamble[t] * log_p + (1 - gamble[t]) * log1m_exp(log_p);
    y_pred_rdm[t]    = bernoulli_rng(p_gamble);
  }

  for (t in 1:nt_dd) {
    int sid;
    real log_sv_later;
    real log_sv_sooner;
    real log_p;
    real p_later;
    sid           = subid_dd[t];
    log_sv_later  = alph[sid] * log(amount_later[t])  - log(1 + kapp[sid] * delay_later[t]);
    log_sv_sooner = alph[sid] * log(amount_sooner[t]);
    log_p         = (log_sv_later / mu[sid]) - log_sum_exp(log_sv_later / mu[sid], log_sv_sooner / mu[sid]);
    if (log_p >= log(1.0)) log_p = log1m(machine_precision());
    else if (log_p <= log(0.0)) log_p = log(machine_precision());
    p_later        = exp(log_p);
    log_lik_dd[t]  = choice_dd[t] * log_p + (1 - choice_dd[t]) * log1m_exp(log_p);
    y_pred_dd[t]   = bernoulli_rng(p_later);
  }
}
