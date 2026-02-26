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
  real mm;
  real<lower=0> ks;
  real<lower=0> ms;
  vector[ns] kapp_raw;
  vector[ns] mu_raw;
}

transformed parameters {
  vector<lower=0>[ns] kapp;
  vector<lower=0>[ns] mu;
  kapp = exp(km + ks * kapp_raw);
  mu   = exp(mm + ms * mu_raw);
}

model {
  real log_pr[nt];
  km ~ normal(-3.0, 1.0);
  ks ~ lognormal(0.0, 1.0);
  mm ~ normal(0.0, 1.0);
  ms ~ lognormal(0.0, 1.0);
  kapp_raw ~ normal(0.0, 1.0);
  mu_raw   ~ normal(0.0, 1.0);
  for (t in 1:nt) {
    int sid;
    real log_sv_later;
    real log_sv_sooner;
    sid           = subid[t];
    log_sv_later  = (alph[sid] * log(amount_later[t])) - log(1 + kapp[sid] * delay_later[t]);
    log_sv_sooner = alph[sid] * log(amount_sooner[t]);
    log_pr[t]     = (log_sv_later / mu[sid]) - log_sum_exp(log_sv_later / mu[sid], log_sv_sooner / mu[sid]);
  }
  target += utility_log_trials(choice, subid, log_pr, nt);
}

generated quantities {
  vector[nt] log_lik;
  int y_pred[nt];
  for (t in 1:nt) {
    int sid;
    real log_sv_later;
    real log_sv_sooner;
    real log_p;
    real p_later;
    sid           = subid[t];
    log_sv_later  = (alph[sid] * log(amount_later[t])) - log(1 + kapp[sid] * delay_later[t]);
    log_sv_sooner = alph[sid] * log(amount_sooner[t]);
    log_p         = (log_sv_later / mu[sid]) - log_sum_exp(log_sv_later / mu[sid], log_sv_sooner / mu[sid]);
    if (log_p >= log(1.0)) {
      log_p = log1m(machine_precision());
    } else if (log_p <= log(0.0)) {
      log_p = log(machine_precision());
    }
    p_later    = exp(log_p);
    log_lik[t] = choice[t] * log_p + (1 - choice[t]) * log1m_exp(log_p);
    y_pred[t]  = bernoulli_rng(p_later);
  }
}
