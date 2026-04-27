library(loo)
library(ggplot2)
library(dplyr)

dir.create("figs", showWarnings = FALSE)

# ── 0. Model definitions ──────────────────────────────────────────────────────
joint_models <- c(
  "1-Joint-SVD-H", "1-Joint-SVR-H",
  "2-Joint-SVD-H", "2-Joint-SVR-H",
  "3-Joint-SVD-H", "3-Joint-SVR-H"
)

# Separate models (from run_model.R)
sep_dd_models  <- c("1-DD-SVD-H",  "1-DD-SVR-H",
                    "2-DD-SVD-H",  "2-DD-SVR-H",
                    "3-DD-SVD-H",  "3-DD-SVR-H")
sep_rdm_models <- c("1-RDM-SVD-H", "1-RDM-SVR-H",
                    "2-RDM-SVD-H", "2-RDM-SVR-H",
                    "3-RDM-SVD-H", "3-RDM-SVR-H")

joint_models    <- joint_models[file.exists(paste0(joint_models,    "_loo_combined.rds"))]
sep_dd_models   <- sep_dd_models[file.exists(paste0(sep_dd_models,  "_loo.rds"))]
sep_rdm_models  <- sep_rdm_models[file.exists(paste0(sep_rdm_models,"_loo.rds"))]

cat("Joint models found:    ", paste(joint_models,    collapse = ", "), "\n")
cat("Separate DD models:    ", paste(sep_dd_models,   collapse = ", "), "\n")
cat("Separate RDM models:   ", paste(sep_rdm_models,  collapse = ", "), "\n\n")

# ── Helper: load a LOO estimates row ─────────────────────────────────────────
# Returns list(elpd, se, looic) from either a proper loo object or our
# sum-of-LOOs list (which has the same $estimates matrix structure)
get_elpd <- function(loo_obj) {
  list(
    elpd  = loo_obj$estimates["elpd_loo", "Estimate"],
    se    = loo_obj$estimates["elpd_loo", "SE"],
    looic = loo_obj$estimates["looic",    "Estimate"]
  )
}

load_rds_if_exists <- function(path) {
  if (file.exists(path)) readRDS(path) else NULL
}

# ── Helper: significance label ────────────────────────────────────────────────
sig_label <- function(p) {
  ifelse(p < 0.001, "***", ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", "n.s.")))
}

# ── Theme ─────────────────────────────────────────────────────────────────────
theme_cmp <- function() {
  theme_classic(base_size = 11) +
    theme(
      plot.title       = element_text(face = "bold", size = 13),
      plot.subtitle    = element_text(size = 10, color = "grey40"),
      axis.text.x      = element_text(angle = 20, hjust = 1),
      legend.position  = "none"
    )
}

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON 1: Joint SVD vs Joint SVR (combined LOO, per dataset)
# ═══════════════════════════════════════════════════════════════════════════════
cat("\n══════════════════════════════════════════════\n")
cat("COMPARISON 1: Joint SVD vs Joint SVR (combined)\n")
cat("══════════════════════════════════════════════\n")

rows_cmp1 <- list()

for (ds in c("1", "2", "3")) {
  svd_name <- paste0(ds, "-Joint-SVD-H")
  svr_name <- paste0(ds, "-Joint-SVR-H")
  if (!all(c(svd_name, svr_name) %in% joint_models)) next
  
  svd_comb <- get_elpd(readRDS(paste0(svd_name, "_loo_combined.rds")))
  svr_comb <- get_elpd(readRDS(paste0(svr_name, "_loo_combined.rds")))
  
  diff  <- svr_comb$elpd - svd_comb$elpd   # positive = SVR better
  se    <- sqrt(svd_comb$se^2 + svr_comb$se^2)
  z     <- diff / se
  p     <- 2 * pnorm(-abs(z))
  winner <- ifelse(diff > 0, "SVR", "SVD")
  
  cat(sprintf("\nDataset %s:\n", ds))
  cat(sprintf("  SVD elpd_loo = %.1f (SE %.1f)\n", svd_comb$elpd, svd_comb$se))
  cat(sprintf("  SVR elpd_loo = %.1f (SE %.1f)\n", svr_comb$elpd, svr_comb$se))
  cat(sprintf("  Diff (SVR-SVD) = %.1f (SE %.1f), z = %.2f, p = %.4f %s\n",
              diff, se, z, p, sig_label(p)))
  cat(sprintf("  Winner: %s\n", winner))
  
  rows_cmp1[[ds]] <- data.frame(
    dataset = ds, svd_elpd = svd_comb$elpd, svd_se = svd_comb$se,
    svr_elpd = svr_comb$elpd, svr_se = svr_comb$se,
    diff = diff, se = se, z = z, p = p, winner = winner
  )
}

if (length(rows_cmp1) > 0) {
  df_cmp1 <- bind_rows(rows_cmp1)
  write.csv(df_cmp1, "loo_comparison_joint_svd_vs_svr.csv", row.names = FALSE)
  
  # Bar plot: SVD vs SVR elpd by dataset
  df_bar1 <- bind_rows(
    data.frame(dataset = df_cmp1$dataset, model_type = "SVD", elpd = df_cmp1$svd_elpd, se = df_cmp1$svd_se),
    data.frame(dataset = df_cmp1$dataset, model_type = "SVR", elpd = df_cmp1$svr_elpd, se = df_cmp1$svr_se)
  ) %>% mutate(label = paste0("Dataset ", dataset))
  
  # Significance stars: one per dataset, positioned above the taller bar
  df_sig1 <- df_cmp1 %>%
    mutate(
      label    = paste0("Dataset ", dataset),
      stars    = sig_label(p),
      elpd_top = pmin(svd_elpd, svr_elpd) * 1.1
    )
  
  p_bar1 <- ggplot(df_bar1, aes(x = label, y = elpd, fill = model_type)) +
    geom_col(position = position_dodge(0.7), width = 0.6, alpha = 0.85) +
    geom_errorbar(aes(ymin = elpd - se, ymax = elpd + se),
                  position = position_dodge(0.7), width = 0.2) +
    geom_text(data = df_sig1, aes(x = label, y = elpd_top, label = stars),
              inherit.aes = FALSE, size = 5, vjust = 0) +
    scale_fill_manual(values = c("SVD" = "#E63946", "SVR" = "#457B9D")) +
    labs(title = "Joint SVD vs SVR — Combined ELPD by dataset",
         y = "ELPD (LOO, combined)", x = "", fill = "Model") +
    theme_classic(base_size = 11) +
    theme(plot.title = element_text(face = "bold"))
  print(p_bar1)
  ggsave("figs/joint_svd_vs_svr_combined.pdf", plot = p_bar1, width = 7, height = 4)
}

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON 2: Joint SVD vs Joint SVR on RDM only
# ═══════════════════════════════════════════════════════════════════════════════
cat("\n══════════════════════════════════════════════\n")
cat("COMPARISON 2: Joint SVD vs Joint SVR (RDM only)\n")
cat("══════════════════════════════════════════════\n")

rows_cmp2 <- list()

for (ds in c("1", "2", "3")) {
  svd_name <- paste0(ds, "-Joint-SVD-H")
  svr_name <- paste0(ds, "-Joint-SVR-H")
  svd_loo  <- load_rds_if_exists(paste0(svd_name, "_loo_rdm.rds"))
  svr_loo  <- load_rds_if_exists(paste0(svr_name, "_loo_rdm.rds"))
  if (is.null(svd_loo) | is.null(svr_loo)) next
  
  # Use loo_compare for proper loo objects
  cmp   <- loo_compare(list(SVD = svd_loo, SVR = svr_loo))
  diff  <- cmp[2, "elpd_diff"]
  se    <- cmp[2, "se_diff"]
  z     <- diff / se
  p     <- 2 * pnorm(-abs(z))
  winner <- rownames(cmp)[1]
  
  cat(sprintf("\nDataset %s RDM:\n", ds))
  print(cmp)
  cat(sprintf("  z = %.2f, p = %.4f %s  Winner: %s\n", z, p, sig_label(p), winner))
  
  rows_cmp2[[ds]] <- data.frame(
    dataset = ds, elpd_diff = diff, se_diff = se, z = z, p = p, winner = winner
  )
}

if (length(rows_cmp2) > 0) {
  write.csv(bind_rows(rows_cmp2), "loo_comparison_joint_svd_vs_svr_rdm.csv", row.names = FALSE)
}

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON 3: Joint SVD vs Joint SVR on DD only
# ═══════════════════════════════════════════════════════════════════════════════
cat("\n══════════════════════════════════════════════\n")
cat("COMPARISON 3: Joint SVD vs Joint SVR (DD only)\n")
cat("══════════════════════════════════════════════\n")

rows_cmp3 <- list()

for (ds in c("1", "2", "3")) {
  svd_name <- paste0(ds, "-Joint-SVD-H")
  svr_name <- paste0(ds, "-Joint-SVR-H")
  svd_loo  <- load_rds_if_exists(paste0(svd_name, "_loo_dd.rds"))
  svr_loo  <- load_rds_if_exists(paste0(svr_name, "_loo_dd.rds"))
  if (is.null(svd_loo) | is.null(svr_loo)) next
  
  cmp   <- loo_compare(list(SVD = svd_loo, SVR = svr_loo))
  diff  <- cmp[2, "elpd_diff"]
  se    <- cmp[2, "se_diff"]
  z     <- diff / se
  p     <- 2 * pnorm(-abs(z))
  winner <- rownames(cmp)[1]
  
  cat(sprintf("\nDataset %s DD:\n", ds))
  print(cmp)
  cat(sprintf("  z = %.2f, p = %.4f %s  Winner: %s\n", z, p, sig_label(p), winner))
  
  rows_cmp3[[ds]] <- data.frame(
    dataset = ds, elpd_diff = diff, se_diff = se, z = z, p = p, winner = winner
  )
}

if (length(rows_cmp3) > 0) {
  write.csv(bind_rows(rows_cmp3), "loo_comparison_joint_svd_vs_svr_dd.csv", row.names = FALSE)
}

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON 4: Joint model vs separate models (combined elpd)
# Compares joint SVD vs (sep DD-SVD + sep RDM-SVD), same for SVR
# ═══════════════════════════════════════════════════════════════════════════════
cat("\n══════════════════════════════════════════════\n")
cat("COMPARISON 4: Joint vs Separate models\n")
cat("══════════════════════════════════════════════\n")

rows_cmp4 <- list()

for (ds in c("1", "2", "3")) {
  for (mtype in c("SVD", "SVR")) {
    
    joint_name  <- paste0(ds, "-Joint-", mtype, "-H")
    sep_dd_name  <- paste0(ds, "-DD-",   mtype, "-H")
    sep_rdm_name <- paste0(ds, "-RDM-",  mtype, "-H")
    
    joint_comb <- load_rds_if_exists(paste0(joint_name,  "_loo_combined.rds"))
    sep_dd_loo  <- load_rds_if_exists(paste0(sep_dd_name,  "_loo.rds"))
    sep_rdm_loo <- load_rds_if_exists(paste0(sep_rdm_name, "_loo.rds"))
    
    if (is.null(joint_comb) | is.null(sep_dd_loo) | is.null(sep_rdm_loo)) {
      cat(sprintf("Skipping Dataset %s %s (missing files)\n", ds, mtype))
      next
    }
    
    joint_e  <- get_elpd(joint_comb)
    
    sep_dd_e  <- get_elpd(sep_dd_loo)
    sep_rdm_e <- get_elpd(sep_rdm_loo)
    sep_elpd  <- sep_dd_e$elpd + sep_rdm_e$elpd
    sep_se    <- sqrt(sep_dd_e$se^2 + sep_rdm_e$se^2)
    
    diff   <- joint_e$elpd - sep_elpd   # positive = joint better
    se_diff <- sqrt(joint_e$se^2 + sep_se^2)
    z      <- diff / se_diff
    p      <- 2 * pnorm(-abs(z))
    winner <- ifelse(diff > 0, "Joint", "Separate")
    
    cat(sprintf("\nDataset %s %s:\n", ds, mtype))
    cat(sprintf("  Joint elpd_loo    = %.1f (SE %.1f)\n", joint_e$elpd, joint_e$se))
    cat(sprintf("  Separate elpd_loo = %.1f (SE %.1f)  [DD %.1f + RDM %.1f]\n",
                sep_elpd, sep_se, sep_dd_e$elpd, sep_rdm_e$elpd))
    cat(sprintf("  Diff (Joint-Sep)  = %.1f (SE %.1f), z = %.2f, p = %.4f %s\n",
                diff, se_diff, z, p, sig_label(p)))
    cat(sprintf("  Winner: %s\n", winner))
    
    key <- paste0(ds, "_", mtype)
    rows_cmp4[[key]] <- data.frame(
      dataset = ds, model_type = mtype,
      joint_elpd = joint_e$elpd, joint_se = joint_e$se,
      sep_elpd = sep_elpd,       sep_se = sep_se,
      diff = diff, se_diff = se_diff, z = z, p = p, winner = winner
    )
  }
}

if (length(rows_cmp4) > 0) {
  df_cmp4 <- bind_rows(rows_cmp4)
  write.csv(df_cmp4, "loo_comparison_joint_vs_separate.csv", row.names = FALSE)
  
  # Bar plot: joint vs separate elpd
  df_bar4 <- bind_rows(
    df_cmp4 %>% select(dataset, model_type, elpd = joint_elpd, se = joint_se) %>%
      mutate(fit = "Joint"),
    df_cmp4 %>% select(dataset, model_type, elpd = sep_elpd,   se = sep_se) %>%
      mutate(fit = "Separate")
  ) %>%
    mutate(label = paste0("DS", dataset, " ", model_type))
  
  # Significance stars: one per dataset/model_type group
  df_sig4 <- df_cmp4 %>%
    mutate(
      label    = paste0("DS", dataset, " ", model_type),
      stars    = sig_label(p),
      elpd_top = pmin(joint_elpd, sep_elpd) * 1.05
    )
  
  p_bar4 <- ggplot(df_bar4, aes(x = label, y = elpd, fill = fit)) +
    geom_col(position = position_dodge(0.7), width = 0.6, alpha = 0.85) +
    geom_errorbar(aes(ymin = elpd - se, ymax = elpd + se),
                  position = position_dodge(0.7), width = 0.2) +
    geom_text(data = df_sig4, aes(x = label, y = elpd_top, label = stars),
              inherit.aes = FALSE, size = 5, vjust = 1) +
    scale_fill_manual(values = c("Joint" = "#2A9D8F", "Separate" = "#E9C46A")) +
    labs(title = "Joint vs Separate models — Combined ELPD",
         subtitle = "Separate = DD elpd + RDM elpd summed",
         y = "ELPD (LOO, combined)", x = "", fill = "") +
    theme_classic(base_size = 11) +
    theme(plot.title = element_text(face = "bold"),
          axis.text.x = element_text(angle = 20, hjust = 1))
  print(p_bar4)
  ggsave("figs/joint_vs_separate_elpd.pdf", plot = p_bar4, width = 9, height = 5)
}

cat("\n══════════════════════════════════════════════\n")
cat("COMPARISON 5: Separate SVD vs SVR (Task-specific)\n")
cat("══════════════════════════════════════════════\n")

rows_sep_plot <- list()

for (ds in c("1", "2", "3")) {
  for (tk in c("DD", "RDM")) {
    svd_name <- paste0(ds, "-", tk, "-SVD-H")
    svr_name <- paste0(ds, "-", tk, "-SVR-H")
    
    svd_loo <- load_rds_if_exists(paste0(svd_name, "_loo.rds"))
    svr_loo <- load_rds_if_exists(paste0(svr_name, "_loo.rds"))
    
    if (is.null(svd_loo) | is.null(svr_loo)) next
    
    # Extract ELPD and SE
    svd_e <- get_elpd(svd_loo)
    svr_e <- get_elpd(svr_loo)
    
    # Calculate difference for significance stars
    diff <- svr_e$elpd - svd_e$elpd
    se_diff <- sqrt(svd_e$se^2 + svr_e$se^2)
    p_val <- 2 * pnorm(-abs(diff / se_diff))
    
    # Store for plotting
    rows_sep_plot[[paste0(ds, "_", tk)]] <- data.frame(
      dataset = ds, 
      task = tk,
      svd_elpd = svd_e$elpd, svd_se = svd_e$se,
      svr_elpd = svr_e$elpd, svr_se = svr_e$se,
      p = p_val,
      label = paste0("Dataset ", ds)
    )
  }
}

if (length(rows_sep_plot) > 0) {
  df_sep_all <- bind_rows(rows_sep_plot)
  
  # Loop through tasks to create two separate plots
  for (tk in unique(df_sep_all$task)) {
    df_tk <- df_sep_all %>% filter(task == tk)
    
    # Pivot to long format for ggplot (SVD vs SVR side-by-side)
    df_bar_tk <- bind_rows(
      df_tk %>% select(label, elpd = svd_elpd, se = svd_se) %>% mutate(model_type = "SVD"),
      df_tk %>% select(label, elpd = svr_elpd, se = svr_se) %>% mutate(model_type = "SVR")
    )
    
    # Set significance star positions
    df_sig_tk <- df_tk %>%
      mutate(stars = sig_label(p),
             elpd_top = pmin(svd_elpd, svr_elpd) * 1.05)
    
    p_tk <- ggplot(df_bar_tk, aes(x = label, y = elpd, fill = model_type)) +
      geom_col(position = position_dodge(0.7), width = 0.6, alpha = 0.85) +
      geom_errorbar(aes(ymin = elpd - se, ymax = elpd + se),
                    position = position_dodge(0.7), width = 0.2) +
      geom_text(data = df_sig_tk, aes(x = label, y = elpd_top, label = stars),
                inherit.aes = FALSE, size = 5, vjust = 1) +
      scale_fill_manual(values = c("SVD" = "#E63946", "SVR" = "#457B9D")) +
      labs(title = paste("Separate", tk, "Models: SVD vs SVR"),
           y = "ELPD (LOO)", x = "", fill = "Model Type") +
      theme_classic(base_size = 11) +
      theme(plot.title = element_text(face = "bold"))
    
    print(p_tk)
    ggsave(paste0("figs/sep_", tolower(tk), "_svd_vs_svr.pdf"), plot = p_tk, width = 7, height = 4)
  }
}

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
cat("\n══════════════════════════════════════════════\n")
cat("SUMMARY\n")
cat("══════════════════════════════════════════════\n")

cat("\nComparison 1 — Joint SVD vs SVR (combined):\n")
if (exists("df_cmp1")) print(df_cmp1) else cat("Not available\n")

cat("\nComparison 4 — Joint vs Separate:\n")
if (exists("df_cmp4")) print(df_cmp4) else cat("Not available\n")

cat("\n=== Analysis complete ===\n")
cat("CSVs and PDFs saved in current directory / figs/\n")