library(rstan)
library(jsonlite)
library(loo)
library(bayesplot)
library(ggplot2)

# ── 0. Define all models to analyze ──────────────────────────────────────────
# Add or remove model names here as you run more fits
# Format: DATASET-TASK-MODELTYPE-H
dd_models  <- c("1-DD-SVD-H",  "1-DD-SVR-H",
                "2-DD-SVD-H",  "2-DD-SVR-H",
                "3-DD-SVD-H",  "3-DD-SVR-H")

rdm_models <- c("1-RDM-SVD-H", "1-RDM-SVR-H",
                "2-RDM-SVD-H", "2-RDM-SVR-H",
                "3-RDM-SVD-H", "3-RDM-SVR-H")

# Only include models that have fully finished
# _loo.rds is one of the last files saved by run_model.R
all_models <- c(dd_models, rdm_models)
all_models <- all_models[file.exists(paste0(all_models, "_loo.rds"))]
cat("Models found:", paste(all_models, collapse = ", "), "\n\n")

# ── Helper: build model metadata ─────────────────────────────────────────────
get_model_info <- function(model_name) {
  is_svr <- grepl("SVR", model_name)
  is_rdm <- grepl("RDM", model_name)
  list(
    name       = model_name,
    is_svr     = is_svr,
    is_rdm     = is_rdm,
    task       = ifelse(is_rdm, "RDM", "DD"),
    dataset    = sub("-(DD|RDM).*", "", model_name),
    choice_var = ifelse(is_rdm, "gamble", "choice"),
    hyper_pars = if      (is_rdm & !is_svr)  c("am", "gm", "as", "gs")
    else if (is_rdm &  is_svr)  c("am", "mm", "as", "ms")
    else if (!is_rdm & !is_svr) c("km", "gm", "ks", "gs")
    else                        c("km", "mm", "ks", "ms"),
    subj_pars  = if (is_svr) c("kapp", "mu") else {
      if (is_rdm) c("alph", "gamma") else c("kapp", "gamma")
    }
  )
}

# ── Helper: load rds only if file exists ─────────────────────────────────────
load_rds_if_exists <- function(path) {
  if (file.exists(path)) readRDS(path) else NULL
}

# ── Helper: load small files only (summary, loo, waic, data) ─────────────────
# Posterior and y_pred are NOT loaded here to avoid memory issues
# They are loaded one model at a time in the trace/PPC sections below
load_model <- function(model_name) {
  info      <- get_model_info(model_name)
  json_base <- sub("-SV[DR]-H$", "", model_name)
  c(info, list(
    data    = jsonlite::read_json(paste0(json_base, ".json"), simplifyVector = TRUE),
    loo     = load_rds_if_exists(paste0(model_name, "_loo.rds")),
    waic    = load_rds_if_exists(paste0(model_name, "_waic.rds")),
    summary = read.csv(paste0(model_name, "_summary.csv"), row.names = 1)
  ))
}

# ── 1. Load all models (small files only) ────────────────────────────────────
cat("Loading models...\n")
models <- lapply(all_models, load_model)
names(models) <- all_models

# ── 2. Parameter summaries ────────────────────────────────────────────────────
cat("\n=== Parameter Summaries ===\n")
for (m in models) {
  cat("\n--", m$name, "--\n")
  rows <- intersect(m$hyper_pars, rownames(m$summary))
  print(m$summary[rows, c("mean", "se_mean", "sd", "X2.5.", "X25.", "X50.", "X75.", "X97.5.", "n_eff", "Rhat")])
}

# ── 3. Rhat & ESS check ─────────────────────────────────────────────────────────────
cat("\n=== Rhat Warnings (Rhat > 1.011 or < 0.99) ===\n")
for (m in models) {
  bad_rhat <- m$summary[!is.na(m$summary$Rhat) & (m$summary$Rhat > 1.01 | m$summary$Rhat < 0.99), , drop = FALSE]
  if (nrow(bad_rhat) > 0) {
    cat("\n--", m$name, ":", nrow(bad_rhat), "parameters with Rhat > 1.01 or < 0.99 --\n")
    print(bad_rhat[, c("mean", "Rhat", "n_eff")])
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
  if (!is.null(m$loo))  print(m$loo)  else cat("LOO not available\n")
  if (!is.null(m$waic)) print(m$waic) else cat("WAIC not available\n")
}

# ── 5. LOO model comparisons ──────────────────────────────────────────────────
# ── 5. LOO model comparisons with significance tests and bar plots ────────────

cat("\n=== LOO Model Comparisons ===\n")

compare_group <- function(model_names, label) {
  available <- model_names[model_names %in% names(models)]
  available <- available[!sapply(models[available], function(m) is.null(m$loo))]
  if (length(available) < 2) {
    cat("Skipping", label, "(fewer than 2 models with LOO available)\n")
    return(NULL)
  }
  cat("\n--", label, "--\n")
  loo_list <- lapply(models[available], function(m) m$loo)
  result   <- loo_compare(loo_list)
  print(result)
  write.csv(as.data.frame(result), paste0("loo_comparison_", gsub(" ", "_", label), ".csv"))
  
  # Significance: z-test from elpd_diff / se_diff
  res_df <- as.data.frame(result)
  if (nrow(res_df) == 2) {
    elpd_diff <- res_df$elpd_diff[2]
    se_diff   <- res_df$se_diff[2]
    z         <- elpd_diff / se_diff
    p         <- 2 * pnorm(-abs(z))
    cat(sprintf("  ELPD diff = %.2f (SE = %.2f), z = %.3f, p = %.4f\n", elpd_diff, se_diff, z, p))
    cat(sprintf("  Best model: %s\n", rownames(res_df)[1]))
  }
  
  # Bar plot: ELPD_LOO per model with SE bars
  elpd_vals <- sapply(loo_list, function(l) l$estimates["elpd_loo", "Estimate"])
  se_vals   <- sapply(loo_list, function(l) l$estimates["elpd_loo", "SE"])
  
  df_bar <- data.frame(
    model = names(loo_list),
    elpd  = elpd_vals,
    se    = se_vals
  )
  df_bar$model <- factor(df_bar$model, levels = df_bar$model)
  
  # Build significance label
  sig_label <- ""
  if (nrow(res_df) == 2) {
    sig_label <- ifelse(p < 0.001, "***",
                        ifelse(p < 0.01,  "**",
                               ifelse(p < 0.05,  "*", "n.s.")))
  }
  
  p_bar <- ggplot(df_bar, aes(x = model, y = elpd, fill = model)) +
    geom_col(width = 0.6, alpha = 0.8) +
    geom_errorbar(aes(ymin = elpd - se, ymax = elpd + se), width = 0.2) +
    labs(title = paste0(label, "  (", sig_label, ")"),
         y = "ELPD (LOO)", x = "") +
    theme_classic() +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 20, hjust = 1))
  print(p_bar)
  ggsave(paste0("figs/loo_bar_", gsub(" ", "_", label), ".pdf"), plot = p_bar, width = 5, height = 4)
  
  result
}

for (ds in c("1", "2", "3")) {
  compare_group(grep(paste0("^", ds, "-DD"),  all_models, value = TRUE), paste("Dataset", ds, "DD"))
  compare_group(grep(paste0("^", ds, "-RDM"), all_models, value = TRUE), paste("Dataset", ds, "RDM"))
}

# ── 6. Trace plots ────────────────────────────────────────────────────────────
# Loaded one model at a time to avoid memory issues
cat("\nGenerating trace plots...\n")
for (model_name in all_models) {
  
  posterior_file <- paste0(model_name, "_posterior.rds")
  if (!file.exists(posterior_file)) {
    cat("Skipping trace plots for", model_name, "(posterior.rds not available — transfer via scp)\n")
    next
  }
  
  m         <- models[[model_name]]
  posterior <- readRDS(posterior_file)
  
  # Hyperparameters
  p <- mcmc_trace(posterior, pars = m$hyper_pars) +
    ggtitle(paste(model_name, "- Hyperparameter Traces"))
  print(p)
  ggsave(paste0("figs/", model_name, "_trace_hyper.pdf"), plot = p, width = 10, height = 6)
  
  # Subject-level parameters — 6 random subjects, 3 per row
  subj_sample <- sample(1:m$data$ns, 6)
  for (par in m$subj_pars) {
    # Extract indices that actually exist in the posterior
    all_pars    <- grep(paste0("^", par, "\\["), dimnames(posterior)[[3]], value = TRUE)
    pars_vec    <- sample(all_pars, min(6, length(all_pars)))
    
    p <- mcmc_trace(posterior, pars = pars_vec) +
      ggtitle(paste(model_name, "-", par, "Traces (6 random subjects)")) +
      facet_wrap(~ parameter, nrow = 2, ncol = 3, scales = "free_y")
    print(p)
    ggsave(paste0("figs/", model_name, "_trace_", par, ".pdf"), plot = p, width = 12, height = 7)
  }
  
  rm(posterior)
  gc()
  cat("Trace plots printed for", model_name, "\n")
}

# ── 7. Posterior predictive checks ───────────────────────────────────────────
# Loaded one model at a time to avoid memory issues
cat("\nGenerating PPC plots...\n")
for (model_name in all_models) {
  
  ypred_file <- paste0(model_name, "_y_pred.rds")
  if (!file.exists(ypred_file)) {
    cat("Skipping PPC plots for", model_name, "(y_pred.rds not available — transfer via scp)\n")
    next
  }
  
  m      <- models[[model_name]]
  y_pred <- readRDS(ypred_file)
  y_obs  <- m$data[[m$choice_var]]
  
  # Overall choice rate
  p <- ppc_stat(y_obs, y_pred, stat = "mean") +
    ggtitle(paste(model_name, "- PPC: Overall choice rate"))
  print(p)
  
  # By subject
  p <- ppc_stat_grouped(
    y_obs, y_pred,
    group = as.character(m$data$subid),
    stat  = "mean"
  ) + ggtitle(paste(model_name, "- PPC: Choice rate by subject"))
  print(p)
  
  # Task-specific plots
  if (m$is_rdm) {
    p <- ppc_stat_grouped(
      y_obs, y_pred,
      group = as.character(cut(m$data$gain, breaks = 5)),
      stat  = "mean"
    ) + ggtitle(paste(model_name, "- PPC: Gamble rate by gain bin"))
    print(p)
    
    p <- ppc_stat_grouped(
      y_obs, y_pred,
      group = as.character(cut(m$data$cert, breaks = 5)),
      stat  = "mean"
    ) + ggtitle(paste(model_name, "- PPC: Gamble rate by cert bin"))
    print(p)
    
  } else {
    p <- ppc_stat_grouped(
      y_obs, y_pred,
      group = as.character(cut(m$data$delay_later, breaks = 5)),
      stat  = "mean"
    ) + ggtitle(paste(model_name, "- PPC: Choice rate by delay bin"))
    print(p)
    
    p <- ppc_stat_grouped(
      y_obs, y_pred,
      group = as.character(cut(m$data$amount_later / m$data$amount_sooner, breaks = 5)),
      stat  = "mean"
    ) + ggtitle(paste(model_name, "- PPC: Choice rate by amount ratio bin"))
    print(p)
  }
  
  rm(y_pred)
  gc()
  cat("PPC plots printed for", model_name, "\n")
}

cat("\n=== Analysis complete ===\n")