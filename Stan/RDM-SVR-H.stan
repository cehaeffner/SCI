functions {
  real utility_log_trials(int[] gamble, int[] subid, real[] log_pr, int nt) {
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
      llhC = llhC + gamble[t] * log_p + (1 - gamble[t]) * log1m_exp(log_p);
    }
    return llhC;
  }
}

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
  real mm;
  real<lower=0> as;
  real<lower=0> ms;
  vector[ns] alph_raw;
  vector[ns] mu_raw;
}

transformed parameters {
  vector<lower=0>[ns] alph;
  vector<lower=0>[ns] mu;
  alph = exp(am + as * alph_raw);
  mu   = exp(mm + ms * mu_raw);
}

model {
  real log_pr[nt];
  am ~ normal(log(0.65), 1.0);
  mm ~ normal(0.0, 1.0);
  as ~ lognormal(0.0, 1.0);
  ms ~ lognormal(0.0, 1.0);
  alph_raw ~ normal(0.0, 1.0);
  mu_raw   ~ normal(0.0, 1.0);
  for (t in 1:nt) {
    int sid;
    real log_svSafe;
    real log_svGamble;
    sid          = subid[t];
    log_svSafe   = alph[sid] * log(cert[t]);
    log_svGamble = log(pgain[t]) + alph[sid] * log(gain[t]);
    log_pr[t]    = (log_svGamble / mu[sid]) - log_sum_exp(log_svGamble / mu[sid], log_svSafe / mu[sid]);
  }
  target += utility_log_trials(gamble, subid, log_pr, nt);
}

generated quantities {
  vector[nt] log_lik;
  int y_pred[nt];
  for (t in 1:nt) {
    int sid;
    real log_svSafe;
    real log_svGamble;
    real log_p;
    real p_gamble;
    sid          = subid[t];
    log_svSafe   = alph[sid] * log(cert[t]);
    log_svGamble = log(pgain[t]) + alph[sid] * log(gain[t]);
    log_p        = (log_svGamble / mu[sid]) - log_sum_exp(log_svGamble / mu[sid], log_svSafe / mu[sid]);
    if (log_p >= log(1.0)) {
      log_p = log1m(machine_precision());
    } else if (log_p <= log(0.0)) {
      log_p = log(machine_precision());
    }
    p_gamble   = exp(log_p);
    log_lik[t] = gamble[t] * log_p + (1 - gamble[t]) * log1m_exp(log_p);
    y_pred[t]  = bernoulli_rng(p_gamble);
  }
}
