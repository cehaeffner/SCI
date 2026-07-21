functions {
  real utility_log_trials(int[] y, real[] log_pr, int nt) {
    real llhC = 0.0;
    for (t in 1:nt) {
      real log_p;
      if (log_pr[t] >= log(1.0)) {
        log_p = log1m(machine_precision());
      } else if (log_pr[t] <= log(0.0)) {
        log_p = log(machine_precision());
      } else {
        log_p = log_pr[t];
      }
      llhC += y[t] * log_p + (1 - y[t]) * log1m_exp(log_p);
    }
    return llhC;
  }
}
data {
  int<lower=1> ns;

  int<lower=1> nt_g;
  int<lower=1, upper=ns> subid_g[nt_g];
  real<lower=0> gain[nt_g];
  real<lower=0> pgain[nt_g];
  real cert[nt_g];
  real<lower=0> alott[nt_g];
  int<lower=0, upper=1> gamble[nt_g];

  int<lower=1> nt_d;
  int<lower=1, upper=ns> subid_d[nt_d];
  real<lower=0> delay_later[nt_d];
  real<lower=0> amount_later[nt_d];
  real<lower=0> amount_sooner[nt_d];
  int<lower=0, upper=1> choice[nt_d];
}
parameters {
  vector[5] pop_mu;                // am, mm_g, bm, km, mm_d
  vector<lower=0>[5] tau;
  cholesky_factor_corr[5] L;
  matrix[5, ns] z;
}
transformed parameters {
  matrix[ns, 5] theta = (rep_matrix(pop_mu, ns) + diag_pre_multiply(tau, L) * z)';
  vector<lower=0>[ns] alph = exp(theta[, 1]);
  vector<lower=0>[ns] mu_g = exp(theta[, 2]);
  vector[ns]          beta =     theta[, 3];
  vector<lower=0>[ns] kapp = exp(theta[, 4]);
  vector<lower=0>[ns] mu_d = exp(theta[, 5]);
}
model {
  real log_pr_g[nt_g];
  real log_pr_d[nt_d];

  pop_mu[1] ~ normal(log(0.65), 1.0);
  pop_mu[2] ~ normal(0.0, 1.0);
  pop_mu[3] ~ normal(0.65, 1.0);
  pop_mu[4] ~ normal(-3.0, 1.0);
  pop_mu[5] ~ normal(0.0, 1.0);
  tau ~ lognormal(0.0, 1.0);
  L   ~ lkj_corr_cholesky(2);
  to_vector(z) ~ std_normal();

  for (t in 1:nt_g) {
    int s = subid_g[t];
    real adjP = fmin(1.0, fmax(0.001, pgain[t] - beta[s] * (alott[t] / 2.0)));
    real log_svSafe   = alph[s] * log(cert[t]);
    real log_svGamble = log(adjP) + alph[s] * log(gain[t]);
    log_pr_g[t] = (log_svGamble / mu_g[s])
                  - log_sum_exp(log_svGamble / mu_g[s], log_svSafe / mu_g[s]);
  }
  target += utility_log_trials(gamble, log_pr_g, nt_g);

  for (t in 1:nt_d) {
    int s = subid_d[t];
    real log_sv_later  = alph[s] * log(amount_later[t]) - log(1 + kapp[s] * delay_later[t]);
    real log_sv_sooner = alph[s] * log(amount_sooner[t]);
    log_pr_d[t] = (log_sv_later / mu_d[s])
                  - log_sum_exp(log_sv_later / mu_d[s], log_sv_sooner / mu_d[s]);
  }
  target += utility_log_trials(choice, log_pr_d, nt_d);
}
generated quantities {
  matrix[5,5] Omega = multiply_lower_tri_self_transpose(L);
  real rho_noise = Omega[2,5];

  vector[nt_g] log_lik_g;
  vector[nt_d] log_lik_d;
  int y_pred_g[nt_g];
  int y_pred_d[nt_d];

  for (t in 1:nt_g) {
    int s = subid_g[t];
    real adjP = fmin(1.0, fmax(0.001, pgain[t] - beta[s] * (alott[t] / 2.0)));
    real log_svSafe   = alph[s] * log(cert[t]);
    real log_svGamble = log(adjP) + alph[s] * log(gain[t]);
    real log_p = (log_svGamble / mu_g[s])
                 - log_sum_exp(log_svGamble / mu_g[s], log_svSafe / mu_g[s]);
    if (log_p >= log(1.0)) log_p = log1m(machine_precision());
    else if (log_p <= log(0.0)) log_p = log(machine_precision());
    log_lik_g[t] = gamble[t] * log_p + (1 - gamble[t]) * log1m_exp(log_p);
    y_pred_g[t]  = bernoulli_rng(exp(log_p));
  }
  for (t in 1:nt_d) {
    int s = subid_d[t];
    real log_sv_later  = alph[s] * log(amount_later[t]) - log(1 + kapp[s] * delay_later[t]);
    real log_sv_sooner = alph[s] * log(amount_sooner[t]);
    real log_p = (log_sv_later / mu_d[s])
                 - log_sum_exp(log_sv_later / mu_d[s], log_sv_sooner / mu_d[s]);
    if (log_p >= log(1.0)) log_p = log1m(machine_precision());
    else if (log_p <= log(0.0)) log_p = log(machine_precision());
    log_lik_d[t] = choice[t] * log_p + (1 - choice[t]) * log1m_exp(log_p);
    y_pred_d[t]  = bernoulli_rng(exp(log_p));
  }
}