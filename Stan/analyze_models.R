library(rstan)
library(jsonlite)
library(loo)
library(bayesplot)
library(ggplot2)

# ── 0. Define all models to analyze ──────────────────────────────────────────
# Add or remove model names here as you run more fits
# Format: DATASET-TASK-MODELTYPE-H
dd_models  <- c("1-DD-SVD-H",  "1-DD-SVR-H",
                "2-DD-SVD-H", "2-DD-SVR-H",
                "3-DD-SVD-H",  "3-DD-SVR-H")

rdm_models <- c("1-RDM-SVD-H",  "1-RDM-SVR-H",
                "2-RDM-SVD-H", "2-RDM-SVR-H",
                "3-RDM-SVD-H",  "3-RDM-SVR-H")

# Only include models whose output files actually exist
all_models <- c(dd_models, rdm_models)
all_models <- all_models[file.exists(paste0(all_models, "_summary.csv"))]
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
    dataset    = sub("-(DD|RDM).*", "", model_name),       # 1, 2, 3
    choice_var = ifelse(is_rdm, "gamble", "choice"),
    hyper_pars = if      (is_rdm & !is_svr) c("am", "gm", "as", "gs")
    else if (is_rdm &  is_svr) c("am", "mm", "as", "ms")
    else if (!is_rdm & !is_svr) c("km", "gm", "ks", "gs")
    else                        c("km", "mm", "ks", "ms"),
    subj_pars  = if (is_svr) c("kapp", "mu") else {
      if (is_rdm) c("alph", "gamma") else c("kapp", "gamma")
    }
  )
}

# ── Helper: load all output files for a model ────────────────────────────────
load_model <- function(model_name) {
  info <- get_model_info(model_name)
  json_base <- sub("-SV[DR]-H$", "", model_name)   # e.g. 1-DD-SVD-H -> 1-DD
  c(info, list(
    data      = jsonlite::read_json(paste0(json_base, ".json"), simplifyVector = TRUE),
    posterior = readRDS(paste0(model_name, "_posterior.rds")),
    log_lik   = readRDS(paste0(model_name, "_log_lik.rds")),
    y_pred    = readRDS(paste0(model_name, "_y_pred.rds")),
    loo       = readRDS(paste0(model_name, "_loo.rds")),
    waic      = readRDS(paste0(model_name, "_waic.rds")),
    summary   = read.csv(paste0(model_name, "_summary.csv"), row.names = 1)
  ))
}

# ── 1. Load all models ────────────────────────────────────────────────────────
cat("Loading models...\n")
models <- lapply(all_models, load_model)
names(models) <- all_models

# ── 2. Parameter summaries ────────────────────────────────────────────────────
cat("\n=== Parameter Summaries ===\n")
for (m in models) {
  cat("\n--", m$name, "--\n")
  rows <- intersect(m$hyper_pars, rownames(m$summary))
  print(m$summary[rows, c("mean", "sd", "2.5%", "50%", "97.5%", "Rhat", "n_eff")])
}

# ── 3. Rhat check ─────────────────────────────────────────────────────────────
cat("\n=== Rhat Warnings (Rhat > 1.01) ===\n")
for (m in models) {
  bad_rhat <- m$summary[!is.na(m$summary$Rhat) & m$summary$Rhat > 1.01, , drop = FALSE]
  if (nrow(bad_rhat) > 0) {
    cat("\n--", m$name, ":", nrow(bad_rhat), "parameters with Rhat > 1.01 --\n")
    print(bad_rhat[, c("mean", "Rhat", "n_eff")])
  } else {
    cat("--", m$name, ": all Rhat < 1.01 (good)\n")
  }
}

# ── 4. LOO and WAIC per model ─────────────────────────────────────────────────
cat("\n=== LOO and WAIC per model ===\n")
for (m in models) {
  cat("\n--", m$name, "--\n")
  print(m$loo)
  print(m$waic)
}

# ── 5. LOO model comparisons ──────────────────────────────────────────────────
cat("\n=== LOO Model Comparisons ===\n")

compare_group <- function(model_names, label) {
  available <- model_names[model_names %in% names(models)]
  if (length(available) < 2) {
    cat("Skipping", label, "(fewer than 2 models available)\n")
    return(NULL)
  }
  cat("\n--", label, "--\n")
  loo_list <- lapply(models[available], function(m) m$loo)
  result   <- loo_compare(loo_list)
  print(result)
  write.csv(as.data.frame(result), paste0("loo_comparison_", gsub(" ", "_", label), ".csv"))
  result
}

# All DD models together
compare_group(dd_models,  "All DD models")

# All RDM models together
compare_group(rdm_models, "All RDM models")

# Within each dataset and task
for (ds in c("1", "2", "3")) {
  compare_group(grep(paste0("^", ds, "-DD"),  all_models, value = TRUE), paste(ds, "DD"))
  compare_group(grep(paste0("^", ds, "-RDM"), all_models, value = TRUE), paste(ds, "RDM"))
}

# ── 6. Trace plots ────────────────────────────────────────────────────────────
cat("\nGenerating trace plots...\n")
for (m in models) {
  
  # Hyperparameters
  p <- mcmc_trace(m$posterior, pars = m$hyper_pars) +
    ggtitle(paste(m$name, "- Hyperparameter Traces"))
  ggsave(paste0(m$name, "_trace_hyper.png"), p, width = 10, height = 8)
  
  # Subject-level parameters
  for (par in m$subj_pars) {
    pars_vec <- paste0(par, "[", 1:m$data$ns, "]")
    p <- mcmc_trace(m$posterior, pars = pars_vec) +
      ggtitle(paste(m$name, "-", par, "Traces"))
    ggsave(paste0(m$name, "_trace_", par, ".png"), p, width = 10, height = 8)
  }
  
  cat("Trace plots saved for", m$name, "\n")
}

# ── 7. Posterior predictive checks ───────────────────────────────────────────
cat("\nGenerating PPC plots...\n")
for (m in models) {
  y_obs  <- m$data[[m$choice_var]]
  y_pred <- m$y_pred
  
  # Overall choice rate
  p <- ppc_stat(y_obs, y_pred, stat = "mean") +
    ggtitle(paste(m$name, "- PPC: Overall choice rate"))
  ggsave(paste0(m$name, "_ppc_overall.png"), p)
  
  # By subject
  p <- ppc_stat_grouped(
    y_obs, y_pred,
    group = as.character(m$data$subid),
    stat  = "mean"
  ) + ggtitle(paste(m$name, "- PPC: Choice rate by subject"))
  ggsave(paste0(m$name, "_ppc_by_subject.png"), p, width = 10, height = 6)
  
  # Task-specific grouping variable
  if (m$is_rdm) {
    # RDM: group by gain and cert bins
    p <- ppc_stat_grouped(
      y_obs, y_pred,
      group = as.character(cut(m$data$gain, breaks = 5)),
      stat  = "mean"
    ) + ggtitle(paste(m$name, "- PPC: Gamble rate by gain bin"))
    ggsave(paste0(m$name, "_ppc_by_gain.png"), p, width = 10, height = 6)
    
    p <- ppc_stat_grouped(
      y_obs, y_pred,
      group = as.character(cut(m$data$cert, breaks = 5)),
      stat  = "mean"
    ) + ggtitle(paste(m$name, "- PPC: Gamble rate by cert bin"))
    ggsave(paste0(m$name, "_ppc_by_cert.png"), p, width = 10, height = 6)
    
  } else {
    # DD: group by delay and amount ratio bins
    p <- ppc_stat_grouped(
      y_obs, y_pred,
      group = as.character(cut(m$data$delay_later, breaks = 5)),
      stat  = "mean"
    ) + ggtitle(paste(m$name, "- PPC: Choice rate by delay bin"))
    ggsave(paste0(m$name, "_ppc_by_delay.png"), p, width = 10, height = 6)
    
    p <- ppc_stat_grouped(
      y_obs, y_pred,
      group = as.character(cut(m$data$amount_later / m$data$amount_sooner, breaks = 5)),
      stat  = "mean"
    ) + ggtitle(paste(m$name, "- PPC: Choice rate by amount ratio bin"))
    ggsave(paste0(m$name, "_ppc_by_amount_ratio.png"), p, width = 10, height = 6)
  }
  
  cat("PPC plots saved for", m$name, "\n")
}

cat("\n=== Analysis complete ===\n")