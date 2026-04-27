library(rstan)
library(jsonlite)
library(loo)
library(bayesplot)
library(ggplot2)


# ── 0. Define all joint models ────────────────────────────────────────────────
joint_models <- c(
  "1-Joint-SVD-H", "1-Joint-SVR-H",
  "2-Joint-SVD-H", "2-Joint-SVR-H",
  "3-Joint-SVD-H", "3-Joint-SVR-H"
)
joint_models <- joint_models[file.exists(paste0(joint_models, "_summary.csv"))]
cat("Models found:", paste(joint_models, collapse = ", "), "\n\n")

# ── Helper: model metadata ────────────────────────────────────────────────────
get_model_info <- function(model_name) {
  is_svr  <- grepl("SVR", model_name)
  dataset <- sub("-Joint.*", "", model_name)
  is_ds3  <- dataset == "3"
  noise_par <- ifelse(is_svr, "mm", "gm")
  noise_sd  <- ifelse(is_svr, "ms", "gs")
  noise_subj <- ifelse(is_svr, "mu", "gamma")
  hyper_pars <- c(noise_par, noise_sd, "am", "as", "km", "ks")
  subj_pars  <- c(noise_subj, "alph", "kapp")
  if (is_ds3) {
    hyper_pars <- c(hyper_pars, "bm", "bs")
    subj_pars  <- c(subj_pars, "beta")
  }
  list(
    name       = model_name,
    is_svr     = is_svr,
    dataset    = dataset,
    is_ds3     = is_ds3,
    noise_par  = noise_par,
    noise_sd   = noise_sd,
    noise_subj = noise_subj,
    hyper_pars = hyper_pars,
    subj_pars  = subj_pars
  )
}

load_rds_if_exists <- function(path) {
  if (file.exists(path)) readRDS(path) else NULL
}

# ── Helper: load small files only (summary, loo, waic, data) ─────────────────
load_model <- function(model_name) {
  info <- get_model_info(model_name)
  ds   <- info$dataset
  c(info, list(
    rdm_data = jsonlite::read_json(paste0(ds, "-RDM.json"), simplifyVector = TRUE),
    dd_data  = jsonlite::read_json(paste0(ds, "-DD.json"),  simplifyVector = TRUE),
    loo_rdm      = load_rds_if_exists(paste0(model_name, "_loo_rdm.rds")),
    loo_dd       = load_rds_if_exists(paste0(model_name, "_loo_dd.rds")),
    loo_combined = load_rds_if_exists(paste0(model_name, "_loo_combined.rds")),
    waic_rdm     = load_rds_if_exists(paste0(model_name, "_waic_rdm.rds")),
    waic_dd      = load_rds_if_exists(paste0(model_name, "_waic_dd.rds")),
    summary      = read.csv(paste0(model_name, "_summary.csv"), row.names = 1)
  ))
}

# ── 1. Load all models ────────────────────────────────────────────────────────
cat("Loading models...\n")
models <- lapply(joint_models, load_model)
names(models) <- joint_models

# ── 2. Parameter summaries ────────────────────────────────────────────────────
cat("\n=== Parameter Summaries ===\n")
for (m in models) {
  cat("\n--", m$name, "--\n")
  rows <- intersect(m$hyper_pars, rownames(m$summary))
  print(m$summary[rows, c("mean", "se_mean", "sd", "X2.5.", "X50.", "X97.5.", "n_eff", "Rhat")])
}

# ── 3. Rhat check ─────────────────────────────────────────────────────────────
cat("\n=== Rhat Warnings (Rhat > 1.01 or < 0.99) ===\n")
for (m in models) {
  bad <- m$summary[!is.na(m$summary$Rhat) & (m$summary$Rhat > 1.01 | m$summary$Rhat < 0.99), , drop = FALSE]
  if (nrow(bad) > 0) {
    cat("\n--", m$name, ":", nrow(bad), "parameters with Rhat > 1.01 or < 0.99 --\n")
    print(bad[, c("mean", "Rhat", "n_eff")])
  } else {
    cat("--", m$name, ": all Rhat between 0.99 and 1.01 (good)\n")
  }
}

cat("\n=== ESS Warnings (n_eff < 400) ===\n")
# Define your threshold (e.g., 100 per chain)
ess_threshold <- 400 

for (m in models) {
  # Standard RStan summaries use 'n_eff'. 
  # If using CmdStanR or newer posterior package, it might be 'ess_bulk' or 'ess_tail'.
  ess_col <- if("n_eff" %in% colnames(m$summary)) "n_eff" else "ess_bulk"
  
  low_ess <- m$summary[!is.na(m$summary[[ess_col]]) & (m$summary[[ess_col]] < ess_threshold), , drop = FALSE]
  
  if (nrow(low_ess) > 0) {
    cat("\n--", m$name, ":", nrow(low_ess), "parameters with ESS <", ess_threshold, "--\n")
    # Sort by lowest ESS to see the worst offenders first
    low_ess <- low_ess[order(low_ess[[ess_col]]), ]
    
    # Print the worst 10 if the list is huge
    head_rows <- min(10, nrow(low_ess))
    print(low_ess[1:head_rows, c("mean", "Rhat", ess_col)])
    
    if(nrow(low_ess) > 10) cat("... and", nrow(low_ess) - 10, "more.\n")
  } else {
    cat("--", m$name, ": all ESS >", ess_threshold, "(good)\n")
  }
}
# ── 4. LOO and WAIC per model ─────────────────────────────────────────────────
cat("\n=== LOO and WAIC per model ===\n")
for (m in models) {
  cat("\n--", m$name, "--\n")
  cat("  RDM LOO:\n");  if (!is.null(m$loo_rdm))      print(m$loo_rdm)      else cat("  not available\n")
  cat("  DD LOO:\n");   if (!is.null(m$loo_dd))        print(m$loo_dd)       else cat("  not available\n")
  cat("  Combined LOO (sum):\n")
  if (!is.null(m$loo_combined)) {
    cat(sprintf("    elpd_loo = %.1f  SE = %.1f  looic = %.1f\n",
                m$loo_combined$estimates["elpd_loo", "Estimate"],
                m$loo_combined$estimates["elpd_loo", "SE"],
                m$loo_combined$estimates["looic",    "Estimate"]))
  } else {
    cat("  not available\n")
  }
}

# ── 5. Trace plots (load posterior one model at a time) ───────────────────────
cat("\nGenerating trace plots...\n")
for (model_name in joint_models) {
  
  posterior_file <- paste0(model_name, "_posterior.rds")
  if (!file.exists(posterior_file)) {
    cat("Skipping trace plots for", model_name, "(posterior.rds not found)\n")
    next
  }
  
  m         <- models[[model_name]]
  posterior <- readRDS(posterior_file)
  
  # Hyperparameters
  p <- mcmc_trace(posterior, pars = m$hyper_pars) +
    ggtitle(paste(model_name, "- Hyperparameter Traces"))
  print(p)
  ggsave(paste0("figs/", model_name, "_trace_hyper.pdf"), plot = p, width = 10, height = 6)
  
  # Subject-level parameters — 6 random subjects
  for (par in m$subj_pars) {
    all_pars <- grep(paste0("^", par, "\\["), dimnames(posterior)[[3]], value = TRUE)
    if (length(all_pars) == 0) next
    pars_vec <- sample(all_pars, min(6, length(all_pars)))
    p <- mcmc_trace(posterior, pars = pars_vec) +
      ggtitle(paste(model_name, "-", par, "Traces (6 random subjects)")) +
      facet_wrap(~ parameter, nrow = 2, ncol = 3, scales = "free_y")
    print(p)
    ggsave(paste0("figs/", model_name, "_trace_", par, ".pdf"), plot = p, width = 12, height = 7)
  }
  
  rm(posterior); gc()
  cat("Trace plots saved for", model_name, "\n")
}

# ── 6. Posterior predictive checks ───────────────────────────────────────────
cat("\nGenerating PPC plots...\n")
for (model_name in joint_models) {
  
  m <- models[[model_name]]
  
  # ── RDM PPC ──
  ypred_rdm_file <- paste0(model_name, "_y_pred_rdm.rds")
  if (file.exists(ypred_rdm_file)) {
    y_pred_rdm <- readRDS(ypred_rdm_file)
    y_obs_rdm  <- m$rdm_data$gamble
    
    # Subsample draws -- 500 is plenty for PPC, avoids memory issues
    max_draws <- 500
    if (nrow(y_pred_rdm) > max_draws)
      y_pred_rdm <- y_pred_rdm[sample(nrow(y_pred_rdm), max_draws), ]
    
    p <- ppc_stat(y_obs_rdm, y_pred_rdm, stat = "mean") +
      ggtitle(paste(model_name, "- PPC RDM: Overall gamble rate"))
    print(p)
    
    p <- ppc_stat_grouped(y_obs_rdm, y_pred_rdm,
                          group = as.character(m$rdm_data$subid), stat = "mean") +
      ggtitle(paste(model_name, "- PPC RDM: Gamble rate by subject"))
    print(p)
    
    p <- ppc_stat_grouped(y_obs_rdm, y_pred_rdm,
                          group = as.character(cut(m$rdm_data$gain, breaks = 5)), stat = "mean") +
      ggtitle(paste(model_name, "- PPC RDM: Gamble rate by gain bin"))
    print(p)
    
    p <- ppc_stat_grouped(y_obs_rdm, y_pred_rdm,
                          group = as.character(cut(m$rdm_data$cert, breaks = 5)), stat = "mean") +
      ggtitle(paste(model_name, "- PPC RDM: Gamble rate by cert bin"))
    print(p)
    
    rm(y_pred_rdm); gc()
    cat("RDM PPC plots printed for", model_name, "\n")
  } else {
    cat("Skipping RDM PPC for", model_name, "(y_pred_rdm.rds not found)\n")
  }
  
  # ── DD PPC ──
  ypred_dd_file <- paste0(model_name, "_y_pred_dd.rds")
  if (file.exists(ypred_dd_file)) {
    y_pred_dd <- readRDS(ypred_dd_file)
    y_obs_dd  <- m$dd_data$choice
    
    # Subsample draws -- 500 is plenty for PPC, avoids memory issues
    if (nrow(y_pred_dd) > max_draws)
      y_pred_dd <- y_pred_dd[sample(nrow(y_pred_dd), max_draws), ]
    
    p <- ppc_stat(y_obs_dd, y_pred_dd, stat = "mean") +
      ggtitle(paste(model_name, "- PPC DD: Overall choice rate"))
    print(p)
    
    p <- ppc_stat_grouped(y_obs_dd, y_pred_dd,
                          group = as.character(m$dd_data$subid), stat = "mean") +
      ggtitle(paste(model_name, "- PPC DD: Choice rate by subject"))
    print(p)
    
    p <- ppc_stat_grouped(y_obs_dd, y_pred_dd,
                          group = as.character(cut(m$dd_data$delay_later, breaks = 5)), stat = "mean") +
      ggtitle(paste(model_name, "- PPC DD: Choice rate by delay bin"))
    print(p)
    
    p <- ppc_stat_grouped(y_obs_dd, y_pred_dd,
                          group = as.character(cut(m$dd_data$amount_later / m$dd_data$amount_sooner, breaks = 5)),
                          stat = "mean") +
      ggtitle(paste(model_name, "- PPC DD: Choice rate by amount ratio bin"))
    print(p)
    
    rm(y_pred_dd); gc()
    cat("DD PPC plots printed for", model_name, "\n")
  } else {
    cat("Skipping DD PPC for", model_name, "(y_pred_dd.rds not found)\n")
  }
}

cat("\n=== Analysis complete ===\n")