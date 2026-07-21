library(rstan)
library(jsonlite)
library(loo)
library(bayesplot)
library(ggplot2)

dir.create("figs", showWarnings = FALSE)

# ── 0. Define all models to analyze ──────────────────────────────────────────
# Format: DATASET-Joint-MODELTYPE-H
all_models <- c("1-Joint-SVD-H", "1-Joint-SVR-H",
                "2-Joint-SVD-H", "2-Joint-SVR-H",
                "3-Joint-SVD-H", "3-Joint-SVR-H")

# _loo_combined.rds is the last LOO file written by the runner
all_models <- all_models[file.exists(paste0(all_models, "_loo_combined.rds"))]
cat("Models found:", paste(all_models, collapse = ", "), "\n\n")
if (length(all_models) == 0) stop("No completed model fits found.")

# Task metadata — one place to change if you renamed suffixes
TASKS <- list(
  rdm = list(key = "rdm", label = "RDM", choice_var = "gamble"),
  dd  = list(key = "dd",  label = "DD",  choice_var = "choice")
)

# ── Helper: build model metadata ─────────────────────────────────────────────
get_model_info <- function(model_name) {
  is_svr <- grepl("SVR", model_name)
  dataset <- sub("-Joint.*", "", model_name)
  is_ds3  <- dataset == "3"
  K       <- if (is_ds3) 5 else 4
  
  # latent dimension order: alph, noise_rdm, [beta], kapp, noise_dd
  noise_idx <- if (is_ds3) c(2, 5) else c(2, 4)
  dim_names <- if (is_ds3) c("alph", "noise_rdm", "beta", "kapp", "noise_dd")
  else        c("alph", "noise_rdm", "kapp", "noise_dd")
  
  list(
    name      = model_name,
    is_svr    = is_svr,
    dataset   = dataset,
    is_ds3    = is_ds3,
    K         = K,
    noise_idx = noise_idx,
    dim_names = dim_names,
    rho_name  = if (is_svr) "rho_noise" else "rho_gamma",
    hyper_pars = c(paste0("pop_mu[", 1:K, "]"),
                   paste0("tau[",    1:K, "]")),
    subj_pars = if (is_svr) {
      if (is_ds3) c("alph", "mu_rdm", "beta", "kapp", "mu_dd")
      else        c("alph", "mu_rdm", "kapp", "mu_dd")
    } else {
      if (is_ds3) c("alph", "gamma_rdm", "beta", "kapp", "gamma_dd")
      else        c("alph", "gamma_rdm", "kapp", "gamma_dd")
    }
  )
}

load_rds_if_exists <- function(path) if (file.exists(path)) readRDS(path) else NULL

# ── Helper: load small files only ────────────────────────────────────────────
# Posterior and y_pred are NOT loaded here — one model at a time, later
load_model <- function(model_name) {
  info <- get_model_info(model_name)
  ds   <- info$dataset
  c(info, list(
    data_rdm     = jsonlite::read_json(paste0(ds, "-RDM.json"), simplifyVector = TRUE),
    data_dd      = jsonlite::read_json(paste0(ds, "-DD.json"),  simplifyVector = TRUE),
    loo_rdm      = load_rds_if_exists(paste0(model_name, "_loo_rdm.rds")),
    loo_dd       = load_rds_if_exists(paste0(model_name, "_loo_dd.rds")),
    loo_combined = load_rds_if_exists(paste0(model_name, "_loo_combined.rds")),
    waic_rdm     = load_rds_if_exists(paste0(model_name, "_waic_rdm.rds")),
    waic_dd      = load_rds_if_exists(paste0(model_name, "_waic_dd.rds")),
    summary      = read.csv(paste0(model_name, "_summary.csv"), row.names = 1)
  ))
}

cat("Loading models...\n")
models <- lapply(all_models, load_model)
names(models) <- all_models

# ── 1. THE HEADLINE: cross-task noise correlation ────────────────────────────
cat("\n=== Cross-task noise-parameter correlation ===\n")
cols <- c("mean", "sd", "X2.5.", "X50.", "X97.5.", "n_eff", "Rhat")

rho_tab <- do.call(rbind, lapply(models, function(m) {
  if (!(m$rho_name %in% rownames(m$summary))) return(NULL)
  r <- m$summary[m$rho_name, intersect(cols, colnames(m$summary))]
  data.frame(model = m$name, scale = ifelse(m$is_svr, "temperature", "inv. temperature"), r)
}))
print(rho_tab, row.names = FALSE)
write.csv(rho_tab, "joint_rho_summary.csv", row.names = FALSE)

# Forest plot of rho across models
if (!is.null(rho_tab) && nrow(rho_tab) > 0) {
  rho_tab$model <- factor(rho_tab$model, levels = rev(rho_tab$model))
  p_rho <- ggplot(rho_tab, aes(x = mean, y = model)) +
    geom_vline(xintercept = 0, linetype = 2, colour = "grey50") +
    geom_errorbarh(aes(xmin = X2.5., xmax = X97.5.), height = 0.15) +
    geom_point(size = 2.5) +
    coord_cartesian(xlim = c(-1, 1)) +
    labs(title = "Cross-task noise correlation (95% CrI)",
         x = expression(rho), y = "") +
    theme_classic()
  print(p_rho)
  ggsave("figs/joint_rho_forest.pdf", p_rho, width = 6, height = 4)
}

# ── 2. Full correlation matrices ─────────────────────────────────────────────
cat("\n=== Full latent correlation matrices ===\n")
for (m in models) {
  cat("\n--", m$name, "--\n")
  om_rows <- grep("^Omega\\[", rownames(m$summary), value = TRUE)
  if (length(om_rows) == 0) { cat("Omega not in summary\n"); next }
  
  idx <- do.call(rbind, lapply(strsplit(gsub("Omega\\[|\\]", "", om_rows), ","),
                               as.numeric))
  M <- matrix(NA, m$K, m$K, dimnames = list(m$dim_names, m$dim_names))
  M[idx] <- m$summary[om_rows, "mean"]
  print(round(M, 3))
  write.csv(M, paste0(m$name, "_Omega_labelled.csv"))
}

# ── 3. Parameter summaries ────────────────────────────────────────────────────
cat("\n=== Population parameter summaries ===\n")
for (m in models) {
  cat("\n--", m$name, "--\n")
  rows <- intersect(c(m$hyper_pars, m$rho_name), rownames(m$summary))
  avail <- intersect(c("mean","se_mean","sd","X2.5.","X25.","X50.","X75.","X97.5.","n_eff","Rhat"),
                     colnames(m$summary))
  labelled <- m$summary[rows, avail]
  rownames(labelled)[seq_along(m$dim_names)] <-
    paste0("mu_", m$dim_names)
  rownames(labelled)[m$K + seq_along(m$dim_names)] <-
    paste0("tau_", m$dim_names)
  print(labelled)
}

# ── 4. Rhat & ESS ─────────────────────────────────────────────────────────────
cat("\n=== Rhat Warnings (Rhat > 1.01 or < 0.99) ===\n")
for (m in models) {
  bad <- m$summary[!is.na(m$summary$Rhat) &
                     (m$summary$Rhat > 1.01 | m$summary$Rhat < 0.99), , drop = FALSE]
  if (nrow(bad) > 0) {
    cat("\n--", m$name, ":", nrow(bad), "parameters --\n")
    print(head(bad[order(-bad$Rhat), c("mean","Rhat","n_eff")], 10))
    if (nrow(bad) > 10) cat("... and", nrow(bad) - 10, "more.\n")
  } else cat("--", m$name, ": all Rhat in [0.99, 1.01] (good)\n")
}

cat("\n=== ESS Warnings (n_eff < 400) ===\n")
ess_threshold <- 400
for (m in models) {
  ess_col <- if ("n_eff" %in% colnames(m$summary)) "n_eff" else "ess_bulk"
  low <- m$summary[!is.na(m$summary[[ess_col]]) &
                     m$summary[[ess_col]] < ess_threshold, , drop = FALSE]
  if (nrow(low) > 0) {
    low <- low[order(low[[ess_col]]), ]
    cat("\n--", m$name, ":", nrow(low), "parameters with ESS <", ess_threshold, "--\n")
    print(low[1:min(10, nrow(low)), c("mean","Rhat",ess_col)])
    if (nrow(low) > 10) cat("... and", nrow(low) - 10, "more.\n")
  } else cat("--", m$name, ": all ESS >", ess_threshold, "(good)\n")
  
  # rho specifically — expected to be the slowest quantity in the fit
  if (m$rho_name %in% rownames(m$summary)) {
    cat(sprintf("   %s: n_eff = %.0f, Rhat = %.3f\n", m$rho_name,
                m$summary[m$rho_name, ess_col], m$summary[m$rho_name, "Rhat"]))
  }
}

# ── 4b. Divergences ──────────────────────────────────────────────────────────
cat("\n=== Divergent transitions ===\n")
for (model_name in all_models) {
  sp <- load_rds_if_exists(paste0(model_name, "_sampler_params.rds"))
  if (is.null(sp)) { cat("--", model_name, ": sampler params not saved\n"); next }
  nd <- sum(sapply(sp, function(x) sum(x[, "divergent__"])))
  nt <- sum(sapply(sp, nrow))
  cat(sprintf("-- %s: %d / %d (%.2f%%)\n", model_name, nd, nt, 100 * nd / nt))
}

# ── 5. LOO and WAIC per model, per task ──────────────────────────────────────
cat("\n=== LOO and WAIC ===\n")
for (m in models) {
  cat("\n--", m$name, "--\n")
  for (tk in TASKS) {
    cat("  [", tk$label, "]\n", sep = "")
    l <- m[[paste0("loo_",  tk$key)]]
    w <- m[[paste0("waic_", tk$key)]]
    if (!is.null(l)) print(l) else cat("  LOO not available\n")
    if (!is.null(w)) print(w) else cat("  WAIC not available\n")
  }
  if (!is.null(m$loo_combined)) {
    cat("  [Combined]\n")
    print(m$loo_combined$estimates)
  }
}

# ── 6. SVD vs SVR comparison, within dataset ─────────────────────────────────
cat("\n=== LOO Model Comparisons (SVD vs SVR) ===\n")

compare_group <- function(model_names, loo_slot, label) {
  available <- model_names[model_names %in% names(models)]
  available <- available[!sapply(models[available],
                                 function(m) is.null(m[[loo_slot]]))]
  if (length(available) < 2) {
    cat("Skipping", label, "(fewer than 2 models available)\n"); return(NULL)
  }
  cat("\n--", label, "--\n")
  loo_list <- lapply(models[available], function(m) m[[loo_slot]])
  result   <- loo_compare(loo_list)
  print(result)
  write.csv(as.data.frame(result),
            paste0("loo_comparison_", gsub(" ", "_", label), ".csv"))
  
  res_df <- as.data.frame(result)
  sig_label <- ""
  if (nrow(res_df) == 2) {
    z <- res_df$elpd_diff[2] / res_df$se_diff[2]
    pv <- 2 * pnorm(-abs(z))
    cat(sprintf("  ELPD diff = %.2f (SE = %.2f), z = %.3f, p = %.4f\n",
                res_df$elpd_diff[2], res_df$se_diff[2], z, pv))
    cat(sprintf("  Best model: %s\n", rownames(res_df)[1]))
    sig_label <- ifelse(pv < 0.001, "***",
                        ifelse(pv < 0.01,  "**",
                               ifelse(pv < 0.05,  "*", "n.s.")))
  }
  
  df_bar <- data.frame(
    model = names(loo_list),
    elpd  = sapply(loo_list, function(l) l$estimates["elpd_loo", "Estimate"]),
    se    = sapply(loo_list, function(l) l$estimates["elpd_loo", "SE"])
  )
  df_bar$model <- factor(df_bar$model, levels = df_bar$model)
  
  p_bar <- ggplot(df_bar, aes(x = model, y = elpd, fill = model)) +
    geom_col(width = 0.6, alpha = 0.8) +
    geom_errorbar(aes(ymin = elpd - se, ymax = elpd + se), width = 0.2) +
    labs(title = paste0(label, "  (", sig_label, ")"), y = "ELPD (LOO)", x = "") +
    theme_classic() +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 20, hjust = 1))
  print(p_bar)
  ggsave(paste0("figs/loo_bar_", gsub(" ", "_", label), ".pdf"),
         p_bar, width = 5, height = 4)
  result
}

for (ds in c("1", "2", "3")) {
  grp <- grep(paste0("^", ds, "-Joint"), all_models, value = TRUE)
  compare_group(grp, "loo_rdm", paste("Dataset", ds, "RDM"))
  compare_group(grp, "loo_dd",  paste("Dataset", ds, "DD"))
}

# ── 7. Trace plots — one model at a time ─────────────────────────────────────
cat("\nGenerating trace plots...\n")
for (model_name in all_models) {
  m <- models[[model_name]]
  
  # 7a. Population level (small file)
  pop_file <- paste0(model_name, "_posterior_pop.rds")
  if (file.exists(pop_file)) {
    post_pop <- readRDS(pop_file)
    avail <- dimnames(post_pop)[[3]]
    pars  <- intersect(c(m$hyper_pars, m$rho_name), avail)
    p <- mcmc_trace(post_pop, pars = pars) +
      ggtitle(paste(model_name, "- Population parameters"))
    print(p)
    ggsave(paste0("figs/", model_name, "_trace_pop.pdf"), p, width = 12, height = 7)
    
    # rho gets a rank plot too — more sensitive than a trace for slow mixing
    if (m$rho_name %in% avail) {
      p <- mcmc_rank_overlay(post_pop, pars = m$rho_name) +
        ggtitle(paste(model_name, "-", m$rho_name, "rank plot"))
      print(p)
      ggsave(paste0("figs/", model_name, "_rank_rho.pdf"), p, width = 6, height = 4)
    }
    rm(post_pop); gc()
  } else cat("Skipping population traces for", model_name, "\n")
  
  # 7b. Subject level (large file)
  subj_file <- paste0(model_name, "_posterior_subj.rds")
  if (file.exists(subj_file)) {
    post_subj <- readRDS(subj_file)
    avail <- dimnames(post_subj)[[3]]
    for (par in m$subj_pars) {
      pool <- grep(paste0("^", par, "\\["), avail, value = TRUE)
      if (length(pool) == 0) next
      pars_vec <- sample(pool, min(6, length(pool)))
      p <- mcmc_trace(post_subj, pars = pars_vec) +
        ggtitle(paste(model_name, "-", par, "(6 random subjects)")) +
        facet_wrap(~ parameter, nrow = 2, ncol = 3, scales = "free_y")
      print(p)
      ggsave(paste0("figs/", model_name, "_trace_", par, ".pdf"),
             p, width = 12, height = 7)
    }
    rm(post_subj); gc()
  } else cat("Skipping subject traces for", model_name, "\n")
  
  # 7c. z — where divergences live in a non-centered parameterization
  z_file <- paste0(model_name, "_posterior_z.rds")
  if (file.exists(z_file)) {
    post_z <- readRDS(z_file)
    pool <- dimnames(post_z)[[3]]
    p <- mcmc_trace(post_z, pars = sample(pool, min(6, length(pool)))) +
      ggtitle(paste(model_name, "- z (sampler scale, 6 random)"))
    print(p)
    ggsave(paste0("figs/", model_name, "_trace_z.pdf"), p, width = 12, height = 7)
    rm(post_z); gc()
  }
  
  cat("Trace plots done for", model_name, "\n")
}

# ── 8. Posterior predictive checks — one model, one task at a time ───────────
cat("\nGenerating PPC plots...\n")
for (model_name in all_models) {
  m <- models[[model_name]]
  
  for (tk in TASKS) {
    yp_file <- paste0(model_name, "_y_pred_", tk$key, ".rds")
    if (!file.exists(yp_file)) {
      cat("Skipping", tk$label, "PPC for", model_name, "\n"); next
    }
    y_pred <- readRDS(yp_file)
    dat    <- m[[paste0("data_", tk$key)]]
    y_obs  <- dat[[tk$choice_var]]
    
    p <- ppc_stat(y_obs, y_pred, stat = "mean") +
      ggtitle(paste(model_name, "-", tk$label, "- Overall choice rate"))
    print(p)
    ggsave(paste0("figs/", model_name, "_ppc_", tk$key, "_overall.pdf"),
           p, width = 6, height = 4)
    
    p <- ppc_stat_grouped(y_obs, y_pred,
                          group = as.character(dat$subid), stat = "mean") +
      ggtitle(paste(model_name, "-", tk$label, "- By subject"))
    print(p)
    ggsave(paste0("figs/", model_name, "_ppc_", tk$key, "_subject.pdf"),
           p, width = 12, height = 9)
    
    if (tk$key == "rdm") {
      for (v in c("gain", "cert")) {
        p <- ppc_stat_grouped(y_obs, y_pred,
                              group = as.character(cut(dat[[v]], breaks = 5)),
                              stat = "mean") +
          ggtitle(paste(model_name, "- RDM - Gamble rate by", v, "bin"))
        print(p)
        ggsave(paste0("figs/", model_name, "_ppc_rdm_", v, ".pdf"),
               p, width = 7, height = 4)
      }
      if (m$is_ds3 && !is.null(dat$alott)) {
        p <- ppc_stat_grouped(y_obs, y_pred,
                              group = as.character(dat$alott), stat = "mean") +
          ggtitle(paste(model_name, "- RDM - Gamble rate by ambiguity level"))
        print(p)
        ggsave(paste0("figs/", model_name, "_ppc_rdm_alott.pdf"),
               p, width = 7, height = 4)
      }
    } else {
      p <- ppc_stat_grouped(y_obs, y_pred,
                            group = as.character(cut(dat$delay_later, breaks = 5)),
                            stat = "mean") +
        ggtitle(paste(model_name, "- DD - Choice rate by delay bin"))
      print(p)
      ggsave(paste0("figs/", model_name, "_ppc_dd_delay.pdf"), p, width = 7, height = 4)
      
      p <- ppc_stat_grouped(y_obs, y_pred,
                            group = as.character(cut(dat$amount_later / dat$amount_sooner,
                                                     breaks = 5)),
                            stat = "mean") +
        ggtitle(paste(model_name, "- DD - Choice rate by amount ratio bin"))
      print(p)
      ggsave(paste0("figs/", model_name, "_ppc_dd_ratio.pdf"), p, width = 7, height = 4)
    }
    
    rm(y_pred); gc()
    cat("PPC done:", model_name, tk$label, "\n")
  }
}

cat("\n=== Analysis complete ===\n")