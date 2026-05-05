library(rstan)
library(jsonlite)
library(loo)

options(mc.cores = 4)
rstan_options(auto_write = TRUE)

# ── 1. Parse model name from command line ─────────────────────────────────────
# Usage: Rscript run_joint_model.R <model_name>
# e.g.   Rscript run_joint_model.R 1-DD-SVD-copula-RDM-SVD
#        Rscript run_joint_model.R 1-DD-SVR-copula-RDM-amb-SVR
args       <- commandArgs(trailingOnly = TRUE)
model_name <- args[1]

if (is.na(model_name)) stop("No model name provided. Usage: Rscript run_joint_model.R <model_name>")
cat("Fitting joint model:", model_name, "\n")

# ── 2. Detect model family ────────────────────────────────────────────────────
is_svr <- grepl("SVR", model_name)
is_amb <- grepl("amb", model_name)

cat("Choice rule:", ifelse(is_svr, "SVR (softmax / mu)", "SVD (logit / gamma)"), "\n")
cat("RDM type:   ", ifelse(is_amb, "with ambiguity (alott + beta)", "risk only"), "\n")

# ── 3. Resolve stan and data files ────────────────────────────────────────────
if (!is_svr & !is_amb) stan_file <- "DD-SVD-copula-RDM-SVD.stan"
if ( is_svr & !is_amb) stan_file <- "DD-SVR-copula-RDM-SVR.stan"
if (!is_svr &  is_amb) stan_file <- "DD-SVD-copula-RDM-amb-SVD.stan"
if ( is_svr &  is_amb) stan_file <- "DD-SVR-copula-RDM-amb-SVR.stan"

dataset  <- sub("-(DD).*", "", model_name)
json_dd  <- paste0(dataset, "-DD.json")
json_rdm <- if (is_amb) paste0(dataset, "-RDM-amb.json") else paste0(dataset, "-RDM.json")

if (!file.exists(stan_file)) stop("Stan file not found: ", stan_file)
if (!file.exists(json_dd))   stop("DD data file not found: ", json_dd)
if (!file.exists(json_rdm))  stop("RDM data file not found: ", json_rdm)

cat("Stan file:   ", stan_file, "\n")
cat("DD data:     ", json_dd,   "\n")
cat("RDM data:    ", json_rdm,  "\n")

# ── 4. Load and merge data ────────────────────────────────────────────────────
dd  <- jsonlite::read_json(json_dd,  simplifyVector = TRUE)
rdm <- jsonlite::read_json(json_rdm, simplifyVector = TRUE)

if (dd$ns != rdm$ns) stop("ns mismatch: DD has ", dd$ns, " subjects, RDM has ", rdm$ns)

# All fields are kept task-specific — nothing is shared.
# Note: alph_dd is passed as fixed data (as in the original DD models).
# RDM alpha (alph) is estimated freely as its own parameter in the RDM block.
data_list <- list(
  ns            = dd$ns,
  # DD fields
  nt_dd         = dd$nt,
  subid_dd      = dd$subid,
  delay_later   = dd$delay_later,
  amount_later  = dd$amount_later,
  amount_sooner = dd$amount_sooner,
  choice        = dd$choice,
  alph_dd       = dd$alph,
  # RDM fields
  nt_rdm        = rdm$nt,
  subid_rdm     = rdm$subid,
  gain          = rdm$gain,
  pgain         = rdm$pgain,
  cert          = rdm$cert,
  gamble        = rdm$gamble
)

if (is_amb) {
  if (is.null(rdm$alott)) stop("alott not found in ", json_rdm, " — wrong data file?")
  data_list$alott <- rdm$alott
}

cat("Data loaded: DD =", data_list$nt_dd, "trials,",
    "RDM =", data_list$nt_rdm, "trials,",
    data_list$ns, "subjects\n")

# ── 5. Fit model ──────────────────────────────────────────────────────────────
fit <- stan(
  file    = stan_file,
  data    = data_list,
  chains  = 4,
  iter    = 11000,
  warmup  = 1000,
  cores   = 4,
  seed    = 42,
  control = list(adapt_delta = 0.95, max_treedepth = 12)
)

# ── 6. Print and save summary ─────────────────────────────────────────────────
# All hyperparameters are fully task-specific — none are shared between tasks.
# The copula only links the raw latent scores; it does not share any parameters.
#
# RDM hyperparams: am, as (alpha); gm_rdm, gs_rdm (gamma) or mm_rdm, ms_rdm (mu)
#                  bm, bs (beta, ambiguity models only)
# DD hyperparams:  km, ks (kappa); gm_dd, gs_dd (gamma) or mm_dd, ms_dd (mu)
#                  note: DD alpha is fixed data, not estimated
# Copula:          rho_out

if (!is_svr) {
  rdm_hyper <- c("am", "as", "gm_rdm", "gs_rdm")
  dd_hyper  <- c("km", "ks", "gm_dd",  "gs_dd")
} else {
  rdm_hyper <- c("am", "as", "mm_rdm", "ms_rdm")
  dd_hyper  <- c("km", "ks", "mm_dd",  "ms_dd")
}
if (is_amb) rdm_hyper <- c(rdm_hyper, "bm", "bs")
hyper_pars <- c(rdm_hyper, dd_hyper, "rho_out")

existing <- intersect(hyper_pars, names(fit))
print(fit, pars = existing)

write.csv(summary(fit)$summary, file = paste0(model_name, "_summary.csv"))
cat("Summary saved.\n")

# ── 7. Extract everything needed before freeing memory ────────────────────────

# 7a. Log-likelihoods — per task and combined for joint LOO
ll_rdm <- extract_log_lik(fit, parameter_name = "log_lik_rdm", merge_chains = FALSE)
ll_dd  <- extract_log_lik(fit, parameter_name = "log_lik_dd",  merge_chains = FALSE)

n_chains <- dim(ll_rdm)[1]
n_draws  <- dim(ll_rdm)[2]

ll_combined <- cbind(
  matrix(ll_rdm, nrow = n_chains * n_draws),
  matrix(ll_dd,  nrow = n_chains * n_draws)
)

rel_eff_rdm      <- relative_eff(exp(ll_rdm), chain_id = rep(1:n_chains, each = n_draws))
rel_eff_dd       <- relative_eff(exp(ll_dd),  chain_id = rep(1:n_chains, each = n_draws))
rel_eff_combined <- relative_eff(exp(ll_combined),
                                  chain_id = rep(1:n_chains, each = n_draws))

# 7b. y_pred — per task
y_pred_rdm <- extract(fit, pars = "y_pred_rdm")$y_pred_rdm
y_pred_dd  <- extract(fit, pars = "y_pred_dd")$y_pred_dd

# 7c. Slim posterior for trace plots
# RDM params: alpha (am/as/alph), choice rule (gm_rdm/gs_rdm/gamma_rdm or mm/ms/mu),
#             beta (bm/bs/beta, ambiguity only)
# DD params:  kappa (km/ks/kapp), choice rule (gm_dd/gs_dd/gamma_dd or mm/ms/mu)
# Copula:     rho_out
if (!is_svr) {
  rdm_pars <- c("am", "as", "gm_rdm", "gs_rdm", "alph", "gamma_rdm")
  dd_pars  <- c("km", "ks", "gm_dd",  "gs_dd",  "kapp", "gamma_dd")
} else {
  rdm_pars <- c("am", "as", "mm_rdm", "ms_rdm", "alph", "mu_rdm")
  dd_pars  <- c("km", "ks", "mm_dd",  "ms_dd",  "kapp", "mu_dd")
}
if (is_amb) rdm_pars <- c(rdm_pars, "bm", "bs", "beta")

pars_to_keep <- intersect(c(rdm_pars, dd_pars, "rho_out"), names(fit))
posterior    <- as.array(fit, pars = pars_to_keep)

# 7d. Rho draws
rho_draws <- extract(fit, pars = "rho_out")$rho_out

# ── 8. Free fit object ────────────────────────────────────────────────────────
rm(fit)
gc()
cat("Fit object freed from memory.\n")

# ── 9. Compute and save LOO and WAIC ─────────────────────────────────────────
loo_joint  <- loo(ll_combined, r_eff = rel_eff_combined)
waic_joint <- waic(ll_combined)
saveRDS(loo_joint,  file = paste0(model_name, "_loo.rds"))
saveRDS(waic_joint, file = paste0(model_name, "_waic.rds"))
print(loo_joint)
print(waic_joint)

# Per-task LOO for direct comparison against the original single-task models
loo_rdm <- loo(ll_rdm, r_eff = rel_eff_rdm)
loo_dd  <- loo(ll_dd,  r_eff = rel_eff_dd)
saveRDS(loo_rdm, file = paste0(model_name, "_loo_rdm.rds"))
saveRDS(loo_dd,  file = paste0(model_name, "_loo_dd.rds"))

cat("LOO and WAIC saved.\n")

rm(ll_rdm, ll_dd, ll_combined,
   rel_eff_rdm, rel_eff_dd, rel_eff_combined,
   loo_joint, waic_joint, loo_rdm, loo_dd)
gc()

# ── 10. Save y_pred ───────────────────────────────────────────────────────────
saveRDS(y_pred_rdm, file = paste0(model_name, "_y_pred_rdm.rds"), compress = TRUE)
saveRDS(y_pred_dd,  file = paste0(model_name, "_y_pred_dd.rds"),  compress = TRUE)
cat("y_pred saved.\n")
rm(y_pred_rdm, y_pred_dd)
gc()

# ── 11. Save posterior and rho ────────────────────────────────────────────────
saveRDS(posterior, file = paste0(model_name, "_posterior.rds"))
saveRDS(rho_draws, file = paste0(model_name, "_rho.rds"))
cat("Posterior and rho saved.\n")

cat("=== Done:", model_name, "===\n")
