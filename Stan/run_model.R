library(rstan)
library(jsonlite)
library(loo)

options(mc.cores = 4)
rstan_options(auto_write = TRUE)

# ── 1. Parse model name from command line ─────────────────────────────────────
# Usage: Rscript run_model.R <model_name>
# e.g.   Rscript run_model.R 1-DD-SVD-H
#        Rscript run_model.R 2-RDM-SVR-H
args       <- commandArgs(trailingOnly = TRUE)
model_name <- args[1]

if (is.na(model_name)) stop("No model name provided. Usage: Rscript run_model.R <model_name>")
cat("Fitting model:", model_name, "\n")

# ── 2. Detect model family ────────────────────────────────────────────────────
is_svr <- grepl("SVR", model_name)
is_rdm <- grepl("RDM", model_name)

cat("Task:       ", ifelse(is_rdm, "RDM (risky decision-making)", "DD (delay discounting)"), "\n")
cat("Model type: ", ifelse(is_svr, "SVR (mu)", "SVD (gamma)"), "\n")

# ── 3. Resolve stan and data files ────────────────────────────────────────────
# Stan files are shared across datasets (1/2/3), named by task and model type only
if (is_rdm & !is_svr)  stan_file <- "RDM-SVD-H.stan"
if (is_rdm &  is_svr)  stan_file <- "RDM-SVR-H.stan"
if (!is_rdm & !is_svr) stan_file <- "DD-SVD-H.stan"
if (!is_rdm &  is_svr) stan_file <- "DD-SVR-H.stan"

# JSON files are shared between SVD and SVR for the same dataset/task
# e.g. 1-DD-SVD-H and 1-DD-SVR-H both load from 1-DD.json
json_base <- sub("-SV[DR]-H$", "", model_name)
json_file <- paste0(json_base, ".json")

if (!file.exists(stan_file)) stop("Stan file not found: ", stan_file)
if (!file.exists(json_file)) stop("JSON file not found: ", json_file)

cat("Stan file:  ", stan_file, "\n")
cat("Data file:  ", json_file, "\n")

# ── 4. Load data ──────────────────────────────────────────────────────────────
data_list <- jsonlite::read_json(json_file, simplifyVector = TRUE)
cat("Data loaded:", data_list$nt, "trials,", data_list$ns, "subjects\n")

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
if (is_rdm & !is_svr)  hyper_pars <- c("am", "gm", "as", "gs")
if (is_rdm &  is_svr)  hyper_pars <- c("am", "mm", "as", "ms")
if (!is_rdm & !is_svr) hyper_pars <- c("km", "gm", "ks", "gs")
if (!is_rdm &  is_svr) hyper_pars <- c("km", "mm", "ks", "ms")

print(fit, pars = hyper_pars)
write.csv(summary(fit)$summary, file = paste0(model_name, "_summary.csv"))
cat("Summary saved.\n")

# ── 7. Extract everything needed from fit before freeing memory ───────────────

# 7a. Extract log_lik for LOO/WAIC
log_lik_mat <- extract_log_lik(fit, parameter_name = "log_lik", merge_chains = FALSE)
rel_eff     <- relative_eff(exp(log_lik_mat))

# 7b. Extract y_pred for PPC plots
y_pred <- extract(fit, pars = "y_pred")$y_pred

# 7c. Extract slim posterior array for trace plots
# Only keeps model parameters — excludes log_lik and y_pred
if (is_rdm & !is_svr)  pars_to_keep <- c("am", "gm", "as", "gs", "alph", "gamma")
if (is_rdm &  is_svr)  pars_to_keep <- c("am", "mm", "as", "ms", "alph", "mu")
if (!is_rdm & !is_svr) pars_to_keep <- c("km", "gm", "ks", "gs", "kapp", "gamma")
if (!is_rdm &  is_svr) pars_to_keep <- c("km", "mm", "ks", "ms", "kapp", "mu")

posterior <- as.array(fit, pars = pars_to_keep)

# ── 8. Free fit object — no longer needed ────────────────────────────────────
rm(fit)
gc()
cat("Fit object freed from memory.\n")

# ── 9. Compute and save LOO and WAIC ─────────────────────────────────────────
loo_result  <- loo(log_lik_mat,  r_eff = rel_eff)
waic_result <- waic(log_lik_mat)
saveRDS(loo_result,  file = paste0(model_name, "_loo.rds"))
saveRDS(waic_result, file = paste0(model_name, "_waic.rds"))
print(loo_result)
print(waic_result)
cat("LOO and WAIC saved.\n")

# Free log_lik matrix — no longer needed
rm(log_lik_mat, rel_eff, loo_result, waic_result)
gc()

# ── 10. Save y_pred ───────────────────────────────────────────────────────────
saveRDS(y_pred, file = paste0(model_name, "_y_pred.rds"), compress = TRUE)
cat("y_pred saved.\n")
rm(y_pred)
gc()

# ── 11. Save slim posterior for trace plots ───────────────────────────────────
saveRDS(posterior, file = paste0(model_name, "_posterior.rds"))
cat("Posterior saved.\n")

cat("=== Done:", model_name, "===\n")