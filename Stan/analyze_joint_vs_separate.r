library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

dir.create("figs", showWarnings = FALSE)

# ── 0. Model definitions ──────────────────────────────────────────────────────
# For each dataset, we correlate:
#   Joint noise (mu/gamma)  vs  RDM noise  AND  DD noise (separate)
#   Joint alph              vs  RDM alph   (separate)
#   Joint kapp              vs  DD kapp    (separate)
#   Joint beta (DS3 only)   vs  RDM beta   (separate)

datasets   <- c("1", "2", "3")
modeltypes <- c("SVD", "SVR")

# ── Helper: extract subject-level posterior means from a summary CSV ──────────
get_subj_means <- function(model_name, par) {
  f <- paste0(model_name, "_summary.csv")
  if (!file.exists(f)) return(NULL)
  df  <- read.csv(f, row.names = 1)
  pat <- paste0("^", par, "\\[\\d+\\]$")
  rows <- grep(pat, rownames(df), value = TRUE)
  if (length(rows) == 0) return(NULL)
  # Extract subject index from row name, return ordered by subject
  idx <- as.integer(sub(paste0(par, "\\[(\\d+)\\]"), "\\1", rows))
  data.frame(subid = idx, value = df[rows, "mean"]) %>%
    arrange(subid) %>%
    pull(value)
}

# ── Helper: one scatter plot with correlation annotation ──────────────────────
scatter_cor <- function(x, y, xlab, ylab, title, colour = "#457B9D") {
  if (is.null(x) || is.null(y) || length(x) != length(y)) return(NULL)
  df    <- data.frame(x = x, y = y)
  ct    <- cor.test(x, y, method = "spearman")
  rho   <- round(ct$estimate, 3)
  pval  <- ct$p.value
  p_label <- ifelse(pval < 0.001, "p < 0.001", sprintf("p = %.3f", pval))
  ann   <- sprintf("rho = %s\n%s", rho, p_label)
  lims  <- range(c(x, y), na.rm = TRUE)
  
  ggplot(df, aes(x = x, y = y)) +
    geom_point(colour = colour, alpha = 0.5, size = 1.5) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                colour = "grey60", linewidth = 0.6) +
    annotate("text", x = -Inf, y = Inf, label = ann,
             hjust = -0.1, vjust = 1.2, size = 3.2, colour = "grey20") +
    coord_equal(xlim = lims, ylim = lims) +
    labs(title = title, x = xlab, y = ylab) +
    theme_classic(base_size = 10) +
    theme(plot.title = element_text(size = 9, face = "bold"))
}

# ── 1. Main loop: one page of plots per dataset × model type ─────────────────
all_results <- list()

for (ds in datasets) {
  for (mtype in modeltypes) {
    
    joint_name   <- paste0(ds, "-Joint-", mtype, "-H")
    sep_rdm_name <- paste0(ds, "-RDM-",   mtype, "-H")
    sep_dd_name  <- paste0(ds, "-DD-",    mtype, "-H")
    noise_subj   <- ifelse(mtype == "SVR", "mu", "gamma")
    is_ds3       <- ds == "3"
    
    # Check that at least joint + one separate model exists
    joint_exists  <- file.exists(paste0(joint_name, "_summary.csv"))
    rdm_exists    <- file.exists(paste0(sep_rdm_name, "_summary.csv"))
    dd_exists     <- file.exists(paste0(sep_dd_name,  "_summary.csv"))
    
    if (!joint_exists) {
      cat("Skipping", joint_name, "(summary not found)\n")
      next
    }
    
    cat("\nProcessing:", joint_name, "\n")
    plots <- list()
    result_rows <- list()
    
    # ── Noise: joint vs RDM separate ──
    if (rdm_exists) {
      j_noise <- get_subj_means(joint_name,   noise_subj)
      r_noise <- get_subj_means(sep_rdm_name, noise_subj)
      if (!is.null(j_noise) && !is.null(r_noise)) {
        p <- scatter_cor(r_noise, j_noise,
                         xlab  = paste0("Separate RDM ", noise_subj),
                         ylab  = paste0("Joint ", noise_subj),
                         title = paste0("Noise (", noise_subj, "): Joint vs RDM"),
                         colour = "#E63946")
        plots[["noise_rdm"]] <- p
        ct <- cor.test(r_noise, j_noise, method = "spearman")
        result_rows[["noise_rdm"]] <- data.frame(
          dataset = ds, model_type = mtype, parameter = noise_subj,
          comparison = "Joint vs RDM", rho = ct$estimate, p = ct$p.value,
          n = length(j_noise)
        )
      }
    }
    
    # ── Noise: joint vs DD separate ──
    if (dd_exists) {
      j_noise <- get_subj_means(joint_name,  noise_subj)
      d_noise <- get_subj_means(sep_dd_name, noise_subj)
      if (!is.null(j_noise) && !is.null(d_noise)) {
        p <- scatter_cor(d_noise, j_noise,
                         xlab  = paste0("Separate DD ", noise_subj),
                         ylab  = paste0("Joint ", noise_subj),
                         title = paste0("Noise (", noise_subj, "): Joint vs DD"),
                         colour = "#F4A261")
        plots[["noise_dd"]] <- p
        ct <- cor.test(d_noise, j_noise, method = "spearman")
        result_rows[["noise_dd"]] <- data.frame(
          dataset = ds, model_type = mtype, parameter = noise_subj,
          comparison = "Joint vs DD", rho = ct$estimate, p = ct$p.value,
          n = length(j_noise)
        )
      }
    }
    
    # ── Alpha: joint vs RDM separate ──
    if (rdm_exists) {
      j_alph <- get_subj_means(joint_name,   "alph")
      r_alph <- get_subj_means(sep_rdm_name, "alph")
      if (!is.null(j_alph) && !is.null(r_alph)) {
        p <- scatter_cor(r_alph, j_alph,
                         xlab  = "Separate RDM alph",
                         ylab  = "Joint alph",
                         title = "Risk preference (alph): Joint vs RDM",
                         colour = "#2A9D8F")
        plots[["alph"]] <- p
        ct <- cor.test(r_alph, j_alph, method = "spearman")
        result_rows[["alph"]] <- data.frame(
          dataset = ds, model_type = mtype, parameter = "alph",
          comparison = "Joint vs RDM", rho = ct$estimate, p = ct$p.value,
          n = length(j_alph)
        )
      }
    }
    
    # ── Kappa: joint vs DD separate ──
    if (dd_exists) {
      j_kapp <- get_subj_means(joint_name,  "kapp")
      d_kapp <- get_subj_means(sep_dd_name, "kapp")
      if (!is.null(j_kapp) && !is.null(d_kapp)) {
        # Use log scale for kappa — heavy right skew otherwise
        p <- scatter_cor(log(d_kapp), log(j_kapp),
                         xlab  = "Separate DD log(kapp)",
                         ylab  = "Joint log(kapp)",
                         title = "Discounting (log kapp): Joint vs DD",
                         colour = "#6A4C93")
        plots[["kapp"]] <- p
        ct <- cor.test(log(d_kapp), log(j_kapp), method = "spearman")
        result_rows[["kapp"]] <- data.frame(
          dataset = ds, model_type = mtype, parameter = "log_kapp",
          comparison = "Joint vs DD", rho = ct$estimate, p = ct$p.value,
          n = length(j_kapp)
        )
      }
    }
    
    # ── Beta (DS3 only): joint vs RDM separate ──
    if (is_ds3 && rdm_exists) {
      j_beta <- get_subj_means(joint_name,   "beta")
      r_beta <- get_subj_means(sep_rdm_name, "beta")
      if (!is.null(j_beta) && !is.null(r_beta)) {
        p <- scatter_cor(r_beta, j_beta,
                         xlab  = "Separate RDM beta",
                         ylab  = "Joint beta",
                         title = "Ambiguity aversion (beta): Joint vs RDM",
                         colour = "#8AC926")
        plots[["beta"]] <- p
        ct <- cor.test(r_beta, j_beta, method = "spearman")
        result_rows[["beta"]] <- data.frame(
          dataset = ds, model_type = mtype, parameter = "beta",
          comparison = "Joint vs RDM", rho = ct$estimate, p = ct$p.value,
          n = length(j_beta)
        )
      }
    }
    
    # ── Assemble page ──
    if (length(plots) > 0) {
      page <- wrap_plots(plots, ncol = 2) +
        plot_annotation(
          title    = paste0(joint_name, " — Parameter correlations: Joint vs Separate"),
          subtitle = "Dashed line = identity (perfect agreement). Solid line = regression fit.",
          theme    = theme(plot.title    = element_text(face = "bold", size = 12),
                           plot.subtitle = element_text(size = 9, colour = "grey40"))
        )
      print(page)
      ggsave(paste0("figs/", joint_name, "_joint_vs_sep_params.pdf"),
             plot = page, width = 10, height = 5 * ceiling(length(plots) / 2))
      cat("Plot saved for", joint_name, "\n")
    }
    
    all_results[[paste0(ds, "_", mtype)]] <- bind_rows(result_rows)
  }
}

# ── 2. Summary table ──────────────────────────────────────────────────────────
cat("\n=== Correlation Summary ===\n")
if (length(all_results) > 0) {
  summary_df <- bind_rows(all_results) %>%
    mutate(
      rho    = round(rho, 3),
      p      = round(p, 4),
      sig    = ifelse(p < 0.001, "***",
                      ifelse(p < 0.01,  "**",
                             ifelse(p < 0.05,  "*", "n.s.")))
    )
  print(summary_df)
  write.csv(summary_df, "joint_vs_separate_correlations.csv", row.names = FALSE)
  cat("\nSaved: joint_vs_separate_correlations.csv\n")
}

# ── 3. Cross-dataset summary plot ────────────────────────────────────────────
if (length(all_results) > 0) {
  
  df_plot <- bind_rows(all_results) %>%
    mutate(
      label     = paste0("DS", dataset, " ", model_type),
      parameter = factor(parameter, levels = c("mu", "gamma", "alph", "log_kapp", "beta"))
    )
  
  p_summary <- ggplot(df_plot, aes(x = label, y = rho, fill = parameter)) +
    geom_col(position = position_dodge(0.7), width = 0.6, alpha = 0.85) +
    geom_hline(yintercept = 0, linewidth = 0.4, colour = "grey50") +
    facet_wrap(~ comparison, ncol = 2) +
    scale_fill_manual(values = c(
      "mu"       = "#F4A261",
      "gamma"    = "#E63946",
      "alph"     = "#2A9D8F",
      "log_kapp" = "#6A4C93",
      "beta"     = "#8AC926"
    )) +
    labs(
      title    = "Joint vs Separate: Spearman correlations by parameter",
      subtitle = "Higher = better agreement between joint and separate estimates",
      x        = "",
      y        = "Spearman rho",
      fill     = "Parameter"
    ) +
    ylim(0, 1) +
    theme_classic(base_size = 11) +
    theme(
      plot.title    = element_text(face = "bold"),
      axis.text.x   = element_text(angle = 20, hjust = 1),
      strip.text    = element_text(face = "bold"),
      legend.position = "right"
    )
  
  print(p_summary)
  ggsave("figs/joint_vs_separate_rho_summary.pdf", plot = p_summary, width = 10, height = 5)
  cat("Summary plot saved.\n")
}

cat("\n=== Done ===\n")