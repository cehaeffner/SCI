library(rstan)
library(jsonlite)
library(loo)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# ── 0. Model name ─────────────────────────────────────────────────────────────
# Format: DATASET-Joint-SVD-H or DATASET-Joint-SVR-H
# Usage: Rscript run_model-coppula.R 1-Joint-SVD-H
#        Rscript run_model-coppula.R 3-Joint-SVR-H
args       <- commandArgs(trailingOnly = TRUE)
model_name <- if (length(args) > 0) args[1] else "1-Joint-SVD-H"
dataset    <- sub("-Joint.*", "", model_name)
is_svr     <- grepl("SVR", model_name)
is_ds3     <- dataset == "3"

cat("Fitting model:", model_name, "\n")
cat("Dataset:      ", dataset, "\n")
cat("Model type:   ", ifelse(is_svr, "SVR (Luce)", "SVD (softmax)"), "\n")
cat("Ambiguity:    ", ifelse(is_ds3, "yes (5 latent dims)", "no (4 latent dims)"), "\n")

# ── 1. Stan file ──────────────────────────────────────────────────────────────
if ( is_svr &  is_ds3) stan_file <- "Joint-SVR-H-coppula-3.stan"
if ( is_svr & !is_ds3) stan_file <- "Joint-SVR-H-coppula.stan"
if (!is_svr &  is_ds3) stan_file <- "Joint-SVD-H-coppula-3.stan"
if (!is_svr & !is_ds3) stan_file <- "Joint-SVD-H-coppula.stan"
cat("Stan file:    ", stan_file, "\n")

if (!file.exists(stan_file)) stop("Stan file not found: ", stan_file)

# ── 2. Load data ──────────────────────────────────────────────────────────────
rdm_file <- paste0(dataset, "-RDM.json")
dd_file  <- paste0(dataset, "-DD.json")
cat("RDM data file:", rdm_file, "\n")
cat("DD data file: ", dd_file,  "\n")

rdm <- read_json(rdm_file, simplifyVector = TRUE)
dd  <- read_json(dd_file,  simplifyVector = TRUE)
cat("RDM data loaded:", rdm$nt, "trials,", rdm$ns, "subjects\n")
cat("DD data loaded: ", dd$nt,  "trials,", dd$ns,  "subjects\n")

if (rdm$ns != dd$ns) stop("RDM and DD datasets have different numbers of subjects!")

# ── 3. Build data list ────────────────────────────────────────────────────────
# Copula Stan files use _g (gambling/RDM) and _d (delay discounting) suffixes
data_list <- list(
  ns            = rdm$ns,
  nt_g          = rdm$nt,
  subid_g       = rdm$subid,
  gain          = rdm$gain,
  pgain         = rdm$pgain,
  cert          = rdm$cert,
  gamble        = rdm$gamble,
  nt_d          = dd$nt,
  subid_d       = dd$subid,
  delay_later   = dd$delay_later,
  amount_later  = dd$amount_later,
  amount_sooner = dd$amount_sooner,
  choice        = dd$choice
)
if (is_ds3) data_list$alott <- rdm$alott

# ── 4. Fit model ──────────────────────────────────────────────────────────────
K <- if (is_ds3) 5 else 4

# pop_mu order (SVD 4D): am, gm_g, km, gm_d
# pop_mu order (SVD 5D): am, gm_g, bm, km, gm_d  (same for SVR with mm)
mu_init <- if (is_ds3) {
  c(log(0.65), 0.0, 0.65, -3.0, 0.0)
} else {
  c(log(0.65), 0.0, -3.0, 0.0)
}

init_fn <- function() list(
  pop_mu = mu_init,
  tau    = rep(0.5, K),
  z      = matrix(0, nrow = K, ncol = data_list$ns)
)

cat("\nStarting sampling...\n")
fit <- stan(
  file    = stan_file,
  data    = data_list,
  chains  = 4,
  iter    = 11000,
  warmup  = 1000,
  cores   = 4,
  seed    = 42,
  init    = init_fn,
  control = list(adapt_delta = 0.95, max_treedepth = 12)
)

# ── 5. Print and save summary ─────────────────────────────────────────────────
rho_name   <- if (is_svr) "rho_noise" else "rho_gamma"
hyper_pars <- c(paste0("pop_mu[", 1:K, "]"), paste0("tau[", 1:K, "]"), rho_name)

cat("\n=== Summary ===\n")
print(summary(fit, pars = hyper_pars)$summary)
cat("\n=== Cross-task noise correlation ===\n")
print(summary(fit, pars = rho_name)$summary)

summ <- as.data.frame(summary(fit)$summary)
write.csv(summ, paste0(model_name, "_summary.csv"))
cat("Summary saved.\n")

# Sampler diagnostics — cheap, save immediately
sampler_diag <- get_sampler_params(fit, inc_warmup = FALSE)
n_div <- sum(sapply(sampler_diag, function(x) sum(x[, "divergent__"])))
cat("Divergent transitions:", n_div, "\n")
saveRDS(sampler_diag, paste0(model_name, "_sampler_params.rds"))
rm(sampler_diag)
print(gc())

# ── 6. Extract and save y_pred ────────────────────────────────────────────────
# Extract and free one at a time to avoid holding both in memory at once
saveRDS(extract(fit, pars = "y_pred_g")$y_pred_g,
        paste0(model_name, "_y_pred_rdm.rds"), compress = TRUE)
gc()
saveRDS(extract(fit, pars = "y_pred_d")$y_pred_d,
        paste0(model_name, "_y_pred_dd.rds"), compress = TRUE)
gc()
cat("y_pred saved.\n")

# ── 7. Extract and save posterior ─────────────────────────────────────────────
# 7a. Population level — small; needed for rho traces and Omega plots
saveRDS(as.array(fit, pars = c("pop_mu", "tau", "Omega", rho_name)),
        paste0(model_name, "_posterior_pop.rds"), compress = TRUE)
gc()

# 7b. Subject level
subj_pars <- if (is_svr) {
  if (is_ds3) c("alph", "mu_g", "beta", "kapp", "mu_d")
  else        c("alph", "mu_g", "kapp", "mu_d")
} else {
  if (is_ds3) c("alph", "gamma_g", "beta", "kapp", "gamma_d")
  else        c("alph", "gamma_g", "kapp", "gamma_d")
}
saveRDS(as.array(fit, pars = subj_pars),
        paste0(model_name, "_posterior_subj.rds"), compress = TRUE)
gc()

# 7c. z — sampler scale; useful for diagnosing divergences in NCP
saveRDS(as.array(fit, pars = "z"),
        paste0(model_name, "_posterior_z.rds"), compress = TRUE)
gc()
cat("Posterior saved.\n")

# ── 8. LOO — RDM (extract, compute, save, free completely) ───────────────────
log_lik_rdm <- extract_log_lik(fit, parameter_name = "log_lik_g", merge_chains = FALSE)
loo_rdm <- loo(log_lik_rdm, moment_match = TRUE, fit = fit, k_threshold = 0.7)    
waic_rdm    <- waic(extract_log_lik(fit, parameter_name = "log_lik_g"))
saveRDS(loo_rdm,  paste0(model_name, "_loo_rdm.rds"))
saveRDS(waic_rdm, paste0(model_name, "_waic_rdm.rds"))
cat("LOO/WAIC RDM saved.\n")
print(loo_rdm)
rm(log_lik_rdm, loo_rdm, waic_rdm)
gc(); gc()

# ── 9. LOO — DD (extract, compute, save, free completely) ────────────────────
log_lik_dd <- extract_log_lik(fit, parameter_name = "log_lik_d", merge_chains = FALSE)
loo_dd <- loo(log_lik_dd, moment_match = TRUE, fit = fit, k_threshold = 0.7)    
waic_dd    <- waic(extract_log_lik(fit, parameter_name = "log_lik_d"))
saveRDS(loo_dd,  paste0(model_name, "_loo_dd.rds"))
saveRDS(waic_dd, paste0(model_name, "_waic_dd.rds"))
cat("LOO/WAIC DD saved.\n")
print(loo_dd)
rm(log_lik_dd, loo_dd, waic_dd)
gc(); gc()

# ── 10. Free fit, then build combined LOO from saved files ────────────────────
# elpd is additive across conditionally independent trial sets,
# so elpd_combined = elpd_rdm + elpd_dd, SE combined in quadrature.
# This avoids cbind-ing the two log_lik matrices into a single giant array.
rm(fit)
gc()

loo_rdm  <- readRDS(paste0(model_name, "_loo_rdm.rds"))
loo_dd   <- readRDS(paste0(model_name, "_loo_dd.rds"))
waic_rdm <- readRDS(paste0(model_name, "_waic_rdm.rds"))
waic_dd  <- readRDS(paste0(model_name, "_waic_dd.rds"))

elpd_combined  <- loo_rdm$estimates["elpd_loo",  "Estimate"] + loo_dd$estimates["elpd_loo",  "Estimate"]
p_combined     <- loo_rdm$estimates["p_loo",      "Estimate"] + loo_dd$estimates["p_loo",      "Estimate"]
se_combined    <- sqrt(loo_rdm$estimates["elpd_loo", "SE"]^2  + loo_dd$estimates["elpd_loo", "SE"]^2)
looic_combined <- -2 * elpd_combined

elpd_waic_combined <- waic_rdm$estimates["elpd_waic", "Estimate"] + waic_dd$estimates["elpd_waic", "Estimate"]
p_waic_combined    <- waic_rdm$estimates["p_waic",    "Estimate"] + waic_dd$estimates["p_waic",    "Estimate"]
se_waic_combined   <- sqrt(waic_rdm$estimates["elpd_waic", "SE"]^2 + waic_dd$estimates["elpd_waic", "SE"]^2)
waic_combined_val  <- -2 * elpd_waic_combined

loo_combined <- list(
  estimates = matrix(
    c(elpd_combined,  se_combined,
      p_combined,     NA,
      looic_combined, 2 * se_combined),
    nrow = 3, ncol = 2, byrow = TRUE,
    dimnames = list(c("elpd_loo", "p_loo", "looic"), c("Estimate", "SE"))
  ),
  method = "loo_sum"
)
waic_combined <- list(
  estimates = matrix(
    c(elpd_waic_combined, se_waic_combined,
      p_waic_combined,    NA,
      waic_combined_val,  2 * se_waic_combined),
    nrow = 3, ncol = 2, byrow = TRUE,
    dimnames = list(c("elpd_waic", "p_waic", "waic"), c("Estimate", "SE"))
  ),
  method = "waic_sum"
)

saveRDS(loo_combined,  paste0(model_name, "_loo_combined.rds"))
saveRDS(waic_combined, paste0(model_name, "_waic_combined.rds"))
cat("LOO/WAIC combined saved.\n")
cat(sprintf("  elpd_loo combined: %.1f (SE %.1f)\n", elpd_combined, se_combined))
cat(sprintf("  looic    combined: %.1f\n", looic_combined))

cat("\n=== Done:", model_name, "===\n")
