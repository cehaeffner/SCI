data {
  int<lower=1> ns;

  // gambling (RDM, with ambiguity)
  int<lower=1> nt_g;
  int<lower=1, upper=ns> subid_g[nt_g];
  real<lower=0> gain[nt_g];
  real<lower=0> pgain[nt_g];
  real cert[nt_g];
  real<lower=0> alott[nt_g];
  int<lower=0, upper=1> gamble[nt_g];

  // delay discounting
  int<lower=1> nt_d;
  int<lower=1, upper=ns> subid_d[nt_d];
  real<lower=0> delay_later[nt_d];
  real<lower=0> amount_later[nt_d];
  real<lower=0> amount_sooner[nt_d];
  int<lower=0, upper=1> choice[nt_d];
}
parameters {
  vector[5] pop_mu;                // am, gm_g, bm, km, gm_d
  vector<lower=0>[5] tau;          // as, gs_g, bs, ks, gs_d
  cholesky_factor_corr[5] L;
  matrix[5, ns] z;
}
transformed parameters {
  matrix[ns, 5] theta = (rep_matrix(pop_mu, ns) + diag_pre_multiply(tau, L) * z)';
  vector<lower=0>[ns] alph    = exp(theta[, 1]);
  vector<lower=0>[ns] gamma_g = exp(theta[, 2]);
  vector[ns]          beta    =     theta[, 3];   // unconstrained
  vector<lower=0>[ns] kapp    = exp(theta[, 4]);
  vector<lower=0>[ns] gamma_d = exp(theta[, 5]);
}
model {
  pop_mu[1] ~ normal(log(0.65), 1.0);
  pop_mu[2] ~ normal(0.0, 1.0);
  pop_mu[3] ~ normal(0.65, 1.0);
  pop_mu[4] ~ normal(-3.0, 1.0);
  pop_mu[5] ~ normal(0.0, 1.0);
  tau ~ lognormal(0.0, 1.0);
  L   ~ lkj_corr_cholesky(2);
  to_vector(z) ~ std_normal();

  {
    vector[nt_g] eta;
    for (t in 1:nt_g) {
      int s = subid_g[t];
      real svSafe = pow(cert[t], alph[s]);
      real adjP   = fmin(1.0, fmax(0.001, pgain[t] - beta[s] * (alott[t] / 2.0)));
      eta[t] = gamma_g[s] * (adjP * pow(gain[t], alph[s]) - svSafe);
    }
    gamble ~ bernoulli_logit(eta);
  }
  {
    vector[nt_d] eta;
    for (t in 1:nt_d) {
      int s = subid_d[t];
      eta[t] = gamma_d[s] * (pow(amount_later[t], alph[s])
                             / (1 + kapp[s] * delay_later[t])
                             - pow(amount_sooner[t], alph[s]));
    }
    choice ~ bernoulli_logit(eta);
  }
}
generated quantities {
  matrix[5,5] Omega = multiply_lower_tri_self_transpose(L);
  real rho_gamma = Omega[2,5];

  vector[nt_g] log_lik_g;
  vector[nt_d] log_lik_d;
  int y_pred_g[nt_g];
  int y_pred_d[nt_d];

  for (t in 1:nt_g) {
    int s = subid_g[t];
    real svSafe = pow(cert[t], alph[s]);
    real adjP   = fmin(1.0, fmax(0.001, pgain[t] - beta[s] * (alott[t] / 2.0)));
    real eta    = gamma_g[s] * (adjP * pow(gain[t], alph[s]) - svSafe);
    log_lik_g[t] = bernoulli_logit_lpmf(gamble[t] | eta);
    y_pred_g[t]  = bernoulli_logit_rng(eta);
  }
  for (t in 1:nt_d) {
    int s = subid_d[t];
    real eta = gamma_d[s] * (pow(amount_later[t], alph[s])
                             / (1 + kapp[s] * delay_later[t])
                             - pow(amount_sooner[t], alph[s]));
    log_lik_d[t] = bernoulli_logit_lpmf(choice[t] | eta);
    y_pred_d[t]  = bernoulli_logit_rng(eta);
  }
}