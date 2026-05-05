library(rstan)
library(jsonlite)
library(loo)
library(bayesplot)
library(ggplot2)

dir.create("figs", showWarnings = FALSE)

# ── 0. Define joint models ────────────────────────────────────────────────────
# Format: DATASET-DD_MODEL-copula-RDM_MODEL
joint_models <- c(
  "1-DD-SVD-copula-RDM-SVD",
  "1-DD-SVR-copula-RDM-SVR",
  "1-DD-SVD-copula-RDM-amb-SVD",
  "1-DD-SVR-copula-RDM-amb-SVR"
  # add further datasets: "2-DD-SVD-copula-RDM-SVD", etc.
)

joint_models <- joint_models[file.exists(paste0(joint_models, "_loo.rds"))]
cat("Models found:", paste(joint_models, collapse = ", "), "\n\n")

# ── Helper: build model metadata ─────────────────────────────────────────────
# All parameters are fully task-specific — nothing is shared.
# The copula only links latent raw scores via rho.
get_model_info <- function(model_name) {
  is_svr  <- grepl("SVR", model_name)
  is_amb  <- grepl("amb", model_name)
  dataset <- sub("-(DD).*", "", model_name)
  list(
    name    = model_name,
    is_svr  = is_svr,
    is_amb  = is_amb,
    dataset = dataset,
    # RDM hyperparams (all RDM-specific)
    rdm_hyper = c(
      "am", "as",
      if (!is_svr) c("gm_rdm", "gs_rdm") else c("mm_rdm", "ms_rdm"),
      if (is_amb)  c("bm", "bs")          else NULL
    ),
    # DD hyperparams (all DD-specific; alpha is fixed data, not estimated)
    dd_hyper = c(
      "km", "ks",
      if (!is_svr) c("gm_dd", "gs_dd") else c("mm_dd", "ms_dd")
    ),
    # Subject-level params by task
    rdm_subj = c(
      "alph",
      if (!is_svr) "gamma_rdm" else "mu_rdm",
      if (is_amb)  "beta"       else NULL
    ),
    dd_subj = c(
      "kapp",
      if (!is_svr) "gamma_dd" else "mu_dd"
    )
  )
}

# ── Helper: load small files ──────────────────────────────────────────────────
load_rds_if_exists <- function(path) if (file.exists(path)) readRDS(path) else NULL

load_model <- function(model_name) {
  info    <- get_model_info(model_name)
  dataset <- info$dataset
  json_dd  <- paste0(dataset, "-DD.json")
  json_rdm <- if (info$is_amb) paste0(dataset, "-RDM-amb.json")
               else             paste0(dataset, "-RDM.json")
  c(info, list(
    data_dd  = if (file.exists(json_dd))  jsonlite::read_json(json_dd,  simplifyVector = TRUE) else NULL,
    data_rdm = if (file.exists(json_rdm)) jsonlite::read_json(json_rdm, simplifyVector = TRUE) else NULL,
    loo      = load_rds_if_exists(paste0(model_name, "_loo.rds")),
    loo_rdm  = load_rds_if_exists(paste0(model_name, "_loo_rdm.rds")),
    loo_dd   = load_rds_if_exists(paste0(model_name, "_loo_dd.rds")),
    waic     = load_rds_if_exists(paste0(model_name, "_waic.rds")),
    summary  = read.csv(paste0(model_name, "_summary.csv"), row.names = 1)
  ))
}

# ── 1. Load models ────────────────────────────────────────────────────────────
cat("Loading models...\n")
models <- lapply(joint_models, load_model)
names(models) <- joint_models

# ── 2. Parameter summaries ────────────────────────────────────────────────────
cat("\n=== Parameter Summaries ===\n")
for (m in models) {
  cat("\n--", m$name, "--\n")
  all_hyper <- c(m$rdm_hyper, m$dd_hyper, "rho_out")
  rows      <- intersect(all_hyper, rownames(m$summary))
  cat("  RDM hyperparameters:\n")
  rdm_rows <- intersect(m$rdm_hyper, rownames(m$summary))
  print(m$summary[rdm_rows, c("mean", "se_mean", "sd", "X2.5.", "X50.", "X97.5.", "n_eff", "Rhat")])
  cat("  DD hyperparameters:\n")
  dd_rows <- intersect(m$dd_hyper, rownames(m$summary))
  print(m$summary[dd_rows,  c("mean", "se_mean", "sd", "X2.5.", "X50.", "X97.5.", "n_eff", "Rhat")])
  cat("  Copula:\n")
  if ("rho_out" %in% rownames(m$summary))
    print(m$summary["rho_out", c("mean", "se_mean", "sd", "X2.5.", "X50.", "X97.5.", "n_eff", "Rhat")])
}

# ── 3. Rho summaries ──────────────────────────────────────────────────────────
cat("\n=== Copula correlation (rho_out) ===\n")
for (m in models) {
  if ("rho_out" %in% rownames(m$summary)) {
    r <- m$summary["rho_out", ]
    cat(sprintf("\n-- %s --\n  rho: mean=%.3f  sd=%.3f  95%% CI [%.3f, %.3f]\n",
                m$name, r$mean, r$sd, r$X2.5., r$X97.5.))
  }
}

# ── 4. Rhat and ESS checks ────────────────────────────────────────────────────
cat("\n=== Rhat Warnings (Rhat > 1.01 or < 0.99) ===\n")
for (m in models) {
  bad <- m$summary[!is.na(m$summary$Rhat) &
                   (m$summary$Rhat > 1.01 | m$summary$Rhat < 0.99), , drop = FALSE]
  if (nrow(bad) > 0) {
    cat("\n--", m$name, ":", nrow(bad), "parameters --\n")
    print(bad[, c("mean", "Rhat", "n_eff")])
  } else {
    cat("--", m$name, ": all Rhat OK\n")
  }
}

cat("\n=== ESS Warnings (n_eff < 400) ===\n")
ess_threshold <- 400
for (m in models) {
  ess_col <- if ("n_eff" %in% colnames(m$summary)) "n_eff" else "ess_bulk"
  low     <- m$summary[!is.na(m$summary[[ess_col]]) &
                        m$summary[[ess_col]] < ess_threshold, , drop = FALSE]
  if (nrow(low) > 0) {
    low <- low[order(low[[ess_col]]), ]
    cat("\n--", m$name, ":", nrow(low), "parameters with ESS <", ess_threshold, "--\n")
    print(head(low, 10)[, c("mean", "Rhat", ess_col)])
    if (nrow(low) > 10) cat("... and", nrow(low) - 10, "more.\n")
  } else {
    cat("--", m$name, ": all ESS OK\n")
  }
}

# ── 5. LOO and WAIC per model ─────────────────────────────────────────────────
cat("\n=== LOO and WAIC per model ===\n")
for (m in models) {
  cat("\n--", m$name, "--\n")
  cat("  Joint LOO (both tasks):\n")
  if (!is.null(m$loo))     print(m$loo)     else cat("  not available\n")
  cat("  Per-task LOO RDM:\n")
  if (!is.null(m$loo_rdm)) print(m$loo_rdm) else cat("  not available\n")
  cat("  Per-task LOO DD:\n")
  if (!is.null(m$loo_dd))  print(m$loo_dd)  else cat("  not available\n")
  cat("  WAIC:\n")
  if (!is.null(m$waic))    print(m$waic)    else cat("  not available\n")
}

# ── 6. LOO model comparisons ──────────────────────────────────────────────────
cat("\n=== LOO Model Comparisons ===\n")

compare_group <- function(model_names, label, loo_slot = "loo") {
  available <- model_names[model_names %in% names(models)]
  available <- available[!sapply(models[available], function(m) is.null(m[[loo_slot]]))]
  if (length(available) < 2) {
    cat("Skipping", label, "(fewer than 2 models)\n"); return(NULL)
  }
  cat("\n--", label, "--\n")
  loo_list <- lapply(models[available], function(m) m[[loo_slot]])
  result   <- loo_compare(loo_list)
  print(result)
  write.csv(as.data.frame(result),
            paste0("loo_comparison_", gsub(" ", "_", label), ".csv"))

  res_df <- as.data.frame(result)
  if (nrow(res_df) == 2) {
    z <- res_df$elpd_diff[2] / res_df$se_diff[2]
    p <- 2 * pnorm(-abs(z))
    cat(sprintf("  ELPD diff = %.2f (SE = %.2f), z = %.3f, p = %.4f\n",
                res_df$elpd_diff[2], res_df$se_diff[2], z, p))
    cat(sprintf("  Best model: %s\n", rownames(res_df)[1]))
  }

  elpd_vals <- sapply(loo_list, function(l) l$estimates["elpd_loo", "Estimate"])
  se_vals   <- sapply(loo_list, function(l) l$estimates["elpd_loo", "SE"])
  df_bar    <- data.frame(model = names(loo_list), elpd = elpd_vals, se = se_vals)
  df_bar$model <- factor(df_bar$model, levels = df_bar$model)

  sig_label <- ""
  if (nrow(res_df) == 2) {
    p_val <- 2 * pnorm(-abs(res_df$elpd_diff[2] / res_df$se_diff[2]))
    sig_label <- ifelse(p_val < 0.001, "***", ifelse(p_val < 0.01, "**",
                 ifelse(p_val < 0.05,  "*",   "n.s.")))
  }

  p_bar <- ggplot(df_bar, aes(x = model, y = elpd, fill = model)) +
    geom_col(width = 0.6, alpha = 0.8) +
    geom_errorbar(aes(ymin = elpd - se, ymax = elpd + se), width = 0.2) +
    labs(title = paste0(label, "  (", sig_label, ")"), y = "ELPD (LOO)", x = "") +
    theme_classic() +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 20, hjust = 1))
  print(p_bar)
  ggsave(paste0("figs/loo_bar_", gsub(" ", "_", label), ".pdf"),
         plot = p_bar, width = 5, height = 4)
  result
}

for (ds in unique(sub("-(DD).*", "", joint_models))) {
  ds_models <- grep(paste0("^", ds, "-"), joint_models, value = TRUE)
  compare_group(ds_models, paste("Dataset", ds, "joint LOO"))
  compare_group(ds_models, paste("Dataset", ds, "RDM LOO"), loo_slot = "loo_rdm")
  compare_group(ds_models, paste("Dataset", ds, "DD LOO"),  loo_slot = "loo_dd")
}

# ── 7. Trace plots ────────────────────────────────────────────────────────────
cat("\nGenerating trace plots...\n")
for (model_name in joint_models) {

  posterior_file <- paste0(model_name, "_posterior.rds")
  if (!file.exists(posterior_file)) {
    cat("Skipping trace plots for", model_name, "(posterior.rds not found)\n"); next
  }

  m         <- models[[model_name]]
  posterior <- readRDS(posterior_file)
  all_pars  <- dimnames(posterior)[[3]]

  # RDM hyperparameters
  rdm_hyper_in_post <- intersect(m$rdm_hyper, all_pars)
  if (length(rdm_hyper_in_post) > 0) {
    p <- mcmc_trace(posterior, pars = rdm_hyper_in_post) +
      ggtitle(paste(model_name, "- RDM Hyperparameter Traces"))
    print(p)
    ggsave(paste0("figs/", model_name, "_trace_rdm_hyper.pdf"), plot = p, width = 10, height = 6)
  }

  # DD hyperparameters
  dd_hyper_in_post <- intersect(m$dd_hyper, all_pars)
  if (length(dd_hyper_in_post) > 0) {
    p <- mcmc_trace(posterior, pars = dd_hyper_in_post) +
      ggtitle(paste(model_name, "- DD Hyperparameter Traces"))
    print(p)
    ggsave(paste0("figs/", model_name, "_trace_dd_hyper.pdf"), plot = p, width = 10, height = 6)
  }

  # Rho
  if ("rho_out" %in% all_pars) {
    p <- mcmc_trace(posterior, pars = "rho_out") +
      ggtitle(paste(model_name, "- Copula rho Trace"))
    print(p)
    ggsave(paste0("figs/", model_name, "_trace_rho.pdf"), plot = p, width = 8, height = 3)
  }

  # Subject-level — 6 random subjects per parameter, per task
  for (par in c(m$rdm_subj, m$dd_subj)) {
    all_subj <- grep(paste0("^", par, "\\["), all_pars, value = TRUE)
    if (length(all_subj) == 0) next
    pars_vec <- sample(all_subj, min(6, length(all_subj)))
    p <- mcmc_trace(posterior, pars = pars_vec) +
      ggtitle(paste(model_name, "-", par, "Traces (6 random subjects)")) +
      facet_wrap(~ parameter, nrow = 2, ncol = 3, scales = "free_y")
    print(p)
    ggsave(paste0("figs/", model_name, "_trace_", par, ".pdf"),
           plot = p, width = 12, height = 7)
  }

  rm(posterior); gc()
  cat("Trace plots saved for", model_name, "\n")
}

# ── 8. Rho posterior plots ────────────────────────────────────────────────────
cat("\nGenerating rho posterior plots...\n")
rho_list <- list()
for (model_name in joint_models) {
  rho_file <- paste0(model_name, "_rho.rds")
  if (!file.exists(rho_file)) next
  rho_list[[model_name]] <- data.frame(rho = readRDS(rho_file), model = model_name)
}

if (length(rho_list) > 0) {
  rho_df <- do.call(rbind, rho_list)
  p_rho  <- ggplot(rho_df, aes(x = rho)) +
    geom_density(fill = "#457B9D", alpha = 0.4, colour = "#1d3557") +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
    facet_wrap(~ model, scales = "free_y") +
    labs(title = "Copula correlation (rho) posterior distributions",
         x = "rho", y = "Density") +
    theme_minimal(base_size = 11)
  print(p_rho)
  ggsave("figs/rho_posteriors.pdf", plot = p_rho, width = 10, height = 6)
  cat("Rho posterior plot saved.\n")
}

# ── 9. Posterior predictive checks ───────────────────────────────────────────
cat("\nGenerating PPC plots...\n")
for (model_name in joint_models) {

  m <- models[[model_name]]

  for (task in c("rdm", "dd")) {

    ypred_file <- paste0(model_name, "_y_pred_", task, ".rds")
    if (!file.exists(ypred_file)) {
      cat("Skipping", task, "PPC for", model_name, "(y_pred not found)\n"); next
    }

    y_pred <- readRDS(ypred_file)
    data   <- if (task == "rdm") m$data_rdm else m$data_dd
    if (is.null(data)) { cat("Data not found for", task, "— skipping\n"); next }

    y_obs      <- if (task == "rdm") data$gamble else data$choice
    task_label <- paste(model_name, toupper(task))

    p <- ppc_stat(y_obs, y_pred, stat = "mean") +
      ggtitle(paste(task_label, "- PPC: Overall choice rate"))
    print(p)

    p <- ppc_stat_grouped(y_obs, y_pred,
                          group = as.character(data$subid), stat = "mean") +
      ggtitle(paste(task_label, "- PPC: Choice rate by subject"))
    print(p)

    if (task == "rdm") {
      p <- ppc_stat_grouped(y_obs, y_pred,
                            group = as.character(cut(data$gain, breaks = 5)),
                            stat = "mean") +
        ggtitle(paste(task_label, "- PPC: Gamble rate by gain bin"))
      print(p)

      p <- ppc_stat_grouped(y_obs, y_pred,
                            group = as.character(cut(data$cert, breaks = 5)),
                            stat = "mean") +
        ggtitle(paste(task_label, "- PPC: Gamble rate by cert bin"))
      print(p)

      if (m$is_amb && !is.null(data$alott)) {
        p <- ppc_stat_grouped(y_obs, y_pred,
                              group = as.character(cut(data$alott, breaks = 5)),
                              stat = "mean") +
          ggtitle(paste(task_label, "- PPC: Gamble rate by ambiguity bin"))
        print(p)
      }

    } else {
      p <- ppc_stat_grouped(y_obs, y_pred,
                            group = as.character(cut(data$delay_later, breaks = 5)),
                            stat = "mean") +
        ggtitle(paste(task_label, "- PPC: Choice rate by delay bin"))
      print(p)

      p <- ppc_stat_grouped(y_obs, y_pred,
                            group = as.character(cut(
                              data$amount_later / data$amount_sooner, breaks = 5)),
                            stat = "mean") +
        ggtitle(paste(task_label, "- PPC: Choice rate by amount ratio bin"))
      print(p)
    }

    rm(y_pred); gc()
    cat("PPC plots printed for", model_name, toupper(task), "\n")
  }
}

cat("\n=== Analysis complete ===\n")
