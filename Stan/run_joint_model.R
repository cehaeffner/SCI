library(rstan)
library(jsonlite)
library(loo)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# ── 0. Model name ─────────────────────────────────────────────────────────────
# Format: DATASET-Joint-SVD-H or DATASET-Joint-SVR-H
# Usage: Rscript run_joint_model.R 1-Joint-SVD-H
args       <- commandArgs(trailingOnly = TRUE)
model_name <- if (length(args) > 0) args[1] else "1-Joint-SVD-H"
dataset    <- sub("-Joint.*", "", model_name)
is_svr     <- grepl("SVR", model_name)
is_ds3     <- dataset == "3"

cat("Fitting model:", model_name, "\n")
cat("Dataset:      ", dataset, "\n")
cat("Model type:   ", ifelse(is_svr, "SVR (Luce)", "SVD (softmax)"), "\n")

# Stan file selection
if ( is_svr &  is_ds3) stan_file <- "Joint-SVR-H-3.stan"
if ( is_svr & !is_ds3) stan_file <- "Joint-SVR-H.stan"
if (!is_svr &  is_ds3) stan_file <- "Joint-SVD-H-3.stan"
if (!is_svr & !is_ds3) stan_file <- "Joint-SVD-H.stan"
cat("Stan file:    ", stan_file, "\n")

# ── 1. Load RDM and DD data ───────────────────────────────────────────────────
rdm_file <- paste0(dataset, "-RDM.json")
dd_file  <- paste0(dataset, "-DD.json")
cat("RDM data file:", rdm_file, "\n")
cat("DD data file: ", dd_file,  "\n")

rdm <- read_json(rdm_file, simplifyVector = TRUE)
dd  <- read_json(dd_file,  simplifyVector = TRUE)
cat("RDM data loaded:", rdm$nt, "trials,", rdm$ns, "subjects\n")
cat("DD data loaded: ", dd$nt,  "trials,", dd$ns,  "subjects\n")

if (rdm$ns != dd$ns) stop("RDM and DD datasets have different numbers of subjects!")

# ── 2. Build joint data list ──────────────────────────────────────────────────
data_list <- list(
  ns            = rdm$ns,
  nt_rdm        = rdm$nt,
  subid_rdm     = rdm$subid,
  gain          = rdm$gain,
  pgain         = rdm$pgain,
  cert          = rdm$cert,
  gamble        = rdm$gamble,
  nt_dd         = dd$nt,
  subid_dd      = dd$subid,
  amount_later  = dd$amount_later,
  amount_sooner = dd$amount_sooner,
  delay_later   = dd$delay_later,
  choice_dd     = dd$choice
)

# Dataset 3 RDM includes ambiguity level
if (is_ds3) data_list$alott <- rdm$alott

# ── 3. Fit model ──────────────────────────────────────────────────────────────
init_fn <- function() {
  base <- list(
    am       = log(0.65),
    as       = 0.5,
    km       = -3.0,
    ks       = 0.5,
    alph_raw = rep(0, data_list$ns),
    kapp_raw = rep(0, data_list$ns)
  )
  if (is_svr) {
    base$mm     <- 0.0
    base$ms     <- 0.5
    base$mu_raw <- rep(0, data_list$ns)
  } else {
    base$gm        <- 0.0
    base$gs        <- 0.5
    base$gamma_raw <- rep(0, data_list$ns)
  }
  if (is_ds3) {
    base$bm       <- 0.65
    base$bs       <- 0.5
    base$beta_raw <- rep(0, data_list$ns)
  }
  base
}

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

# ── 4. Print and save summary ─────────────────────────────────────────────────
noise_par  <- ifelse(is_svr, "mm", "gm")
noise_sd   <- ifelse(is_svr, "ms", "gs")
hyper_pars <- c(noise_par, noise_sd, "am", "as", "km", "ks")
if (is_ds3) hyper_pars <- c(hyper_pars, "bm", "bs")

cat("\n=== Summary ===\n")
print(summary(fit, pars = hyper_pars)$summary)

summ <- as.data.frame(summary(fit)$summary)
write.csv(summ, paste0(model_name, "_summary.csv"))
cat("Summary saved.\n")
print(gc())

# ── 5. LOO — RDM ─────────────────────────────────────────────────────────────
log_lik_rdm <- extract_log_lik(fit, parameter_name = "log_lik_rdm", merge_chains = FALSE)
loo_rdm     <- loo(log_lik_rdm, moment_match = FALSE)
waic_rdm    <- waic(extract_log_lik(fit, parameter_name = "log_lik_rdm"))
saveRDS(loo_rdm,  paste0(model_name, "_loo_rdm.rds"))
saveRDS(waic_rdm, paste0(model_name, "_waic_rdm.rds"))
cat("LOO/WAIC RDM saved.\n")
print(loo_rdm)
rm(log_lik_rdm, loo_rdm, waic_rdm)
gc()

# ── 6. LOO — DD ──────────────────────────────────────────────────────────────
log_lik_dd <- extract_log_lik(fit, parameter_name = "log_lik_dd", merge_chains = FALSE)
loo_dd     <- loo(log_lik_dd, moment_match = FALSE)
waic_dd    <- waic(extract_log_lik(fit, parameter_name = "log_lik_dd"))
saveRDS(loo_dd,  paste0(model_name, "_loo_dd.rds"))
saveRDS(waic_dd, paste0(model_name, "_waic_dd.rds"))
cat("LOO/WAIC DD saved.\n")
print(loo_dd)
rm(log_lik_dd, loo_dd, waic_dd)
gc()

# ── 7. Save y_pred ────────────────────────────────────────────────────────────
saveRDS(extract(fit, pars = "y_pred_rdm")$y_pred_rdm, paste0(model_name, "_y_pred_rdm.rds"), compress = TRUE)
saveRDS(extract(fit, pars = "y_pred_dd")$y_pred_dd,   paste0(model_name, "_y_pred_dd.rds"),  compress = TRUE)
cat("y_pred saved.\n")
gc()

# ── 8. Save posterior ─────────────────────────────────────────────────────────
noise_subj   <- ifelse(is_svr, "mu", "gamma")
pars_to_keep <- c(noise_subj, "alph", "kapp", noise_par, noise_sd, "am", "as", "km", "ks")
if (is_ds3) pars_to_keep <- c(pars_to_keep, "beta", "bm", "bs")

saveRDS(as.array(fit, pars = pars_to_keep), paste0(model_name, "_posterior.rds"), compress = TRUE)
rm(fit)
gc()
cat("Posterior saved.\n")

cat("\n=== Done:", model_name, "===\n")
