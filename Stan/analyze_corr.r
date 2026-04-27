library(jsonlite)
library(ggplot2)
library(patchwork)

# ── Helper: extract subject-level posterior means for a parameter ─────────────
get_subj_means <- function(model_name, par) {
  f <- paste0(model_name, "_summary.csv")
  if (!file.exists(f)) return(NULL)
  s    <- read.csv(f, row.names = 1)
  rows <- grep(paste0("^", par, "\\["), rownames(s))
  if (length(rows) == 0) return(NULL)
  s[rows, "mean"]
}

# ── Define pairs ──────────────────────────────────────────────────────────────
pairs <- list(
  list(dd = "1-DD-SVD-H", rdm = "1-RDM-SVD-H", par = "gamma", cohort = "Cohort 1", model = "SVD"),
  list(dd = "1-DD-SVR-H", rdm = "1-RDM-SVR-H", par = "mu",    cohort = "Cohort 1", model = "SVR"),
  list(dd = "2-DD-SVD-H", rdm = "2-RDM-SVD-H", par = "gamma", cohort = "Cohort 2", model = "SVD"),
  list(dd = "2-DD-SVR-H", rdm = "2-RDM-SVR-H", par = "mu",    cohort = "Cohort 2", model = "SVR"),
  list(dd = "3-DD-SVD-H", rdm = "3-RDM-SVD-H", par = "gamma", cohort = "Cohort 3", model = "SVD"),
  list(dd = "3-DD-SVR-H", rdm = "3-RDM-SVR-H", par = "mu",    cohort = "Cohort 3", model = "SVR")
)

# ── Collect plots into a named list keyed by model/cohort ────────────────────
plot_grid <- list()

for (p in pairs) {
  dd_vals  <- get_subj_means(p$dd,  p$par)
  rdm_vals <- get_subj_means(p$rdm, p$par)
  
  if (is.null(dd_vals) || is.null(rdm_vals)) next
  if (length(dd_vals) != length(rdm_vals))   next
  
  ct       <- cor.test(dd_vals, rdm_vals, method = "spearman")
  ax_range <- range(c(dd_vals, rdm_vals))
  ax_range <- ax_range + c(-0.05, 0.05) * diff(ax_range)
  
  df <- data.frame(dd = dd_vals, rdm = rdm_vals)
  
  col <- if (p$model == "SVR") "magenta" else "steelblue"
  
  pl <- ggplot(df, aes(x = dd, y = rdm)) +
    geom_point(color = col, size = 2, alpha = 0.8) +
    geom_abline(intercept = 0, slope = 1, color = "black", linewidth = 0.8) +
    coord_fixed(xlim = ax_range, ylim = ax_range) +
    annotate("text",
             x     = ax_range[1] + 0.05 * diff(ax_range),
             y     = ax_range[2] - 0.05 * diff(ax_range),
             label = sprintf("r = %.2f\np = %.3f", ct$estimate, ct$p.value),
             hjust = 0, vjust = 1, size = 3.5) +
    labs(
      title = p$cohort,
      x     = paste0(p$par, " (DD)"),
      y     = paste0(p$par, " (RDM)")
    ) +
    theme_classic() +
    theme(aspect.ratio = 1,
          plot.title   = element_text(hjust = 0.5))
  
  plot_grid[[paste0(p$model, "_", p$cohort)]] <- pl
}

# ── Arrange: rows = model (SVR, SVD), columns = cohort (1, 2, 3) ─────────────
svr_plots <- list(
  plot_grid[["SVR_Cohort 1"]],
  plot_grid[["SVR_Cohort 2"]],
  plot_grid[["SVR_Cohort 3"]]
)
svd_plots <- list(
  plot_grid[["SVD_Cohort 1"]],
  plot_grid[["SVD_Cohort 2"]],
  plot_grid[["SVD_Cohort 3"]]
)

# Remove NULLs
svr_plots <- Filter(Negate(is.null), svr_plots)
svd_plots <- Filter(Negate(is.null), svd_plots)

all_plots <- c(svr_plots, svd_plots)

if (length(all_plots) > 0) {
  print(wrap_plots(all_plots, nrow = 2, ncol = 3))
}