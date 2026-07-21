library(ggplot2)
library(dplyr)
library(tidyr)
library(colorspace)

dir.create("figs", showWarnings = FALSE)

# ── 0. Model definitions (mirror analyze_models.R) ──────────────────────────
all_models <- c("1-Joint-SVD-H", "1-Joint-SVR-H",
                "2-Joint-SVD-H", "2-Joint-SVR-H",
                "3-Joint-SVD-H", "3-Joint-SVR-H")

all_models <- all_models[file.exists(paste0(all_models, "_summary.csv"))]
cat("Models found:", paste(all_models, collapse = ", "), "\n\n")
if (length(all_models) == 0) stop("No completed model fits found.")

# ── 1. Model metadata ────────────────────────────────────────────────────────
get_model_info <- function(model_name) {
  is_svr <- grepl("SVR", model_name)
  ds     <- sub("-Joint.*", "", model_name)
  is_ds3 <- ds == "3"
  K      <- if (is_ds3) 5 else 4
  
  noise_rdm <- if (is_svr) "mu_rdm"   else "gamma_rdm"
  noise_dd  <- if (is_svr) "mu_dd"    else "gamma_dd"
  
  subj <- c("alph", noise_rdm, "kapp", noise_dd)
  if (is_ds3) subj <- c(subj, "beta")
  
  list(
    name       = model_name,
    is_svr     = is_svr,
    dataset    = ds,
    is_ds3     = is_ds3,
    K          = K,
    noise_rdm  = noise_rdm,
    noise_dd   = noise_dd,
    rho_name   = if (is_svr) "rho_noise" else "rho_gamma",
    # latent dimension order in theta / Omega
    dim_names  = if (is_ds3) c("alph", noise_rdm, "beta", "kapp", noise_dd)
    else        c("alph", noise_rdm, "kapp", noise_dd),
    hyper_pars = c(paste0("pop_mu[", 1:K, "]"), paste0("tau[", 1:K, "]")),
    subj_pars  = subj
  )
}

# ── 2. Load summaries ────────────────────────────────────────────────────────
load_summary <- function(model_name) {
  info <- get_model_info(model_name)
  df   <- read.csv(paste0(model_name, "_summary.csv"), row.names = 1)
  df$param_name <- rownames(df)
  df$model      <- model_name
  df$is_svr     <- info$is_svr
  df$dataset    <- info$dataset
  
  subj_pat  <- paste0("^(", paste(info$subj_pars,  collapse = "|"), ")\\[")
  hyper_pat <- paste0("^(", paste(gsub("\\[", "\\\\[", info$hyper_pars),
                                  collapse = "|"), ")$")
  df$param_type <- case_when(
    grepl(subj_pat,  df$param_name)  ~ "subject",
    grepl(hyper_pat, df$param_name)  ~ "hyper",
    grepl("^Omega\\[", df$param_name) ~ "corr",
    df$param_name == info$rho_name   ~ "corr",
    TRUE                             ~ "other"
  )
  df$param_base <- sub("\\[.*", "", df$param_name)
  df$subj_idx   <- suppressWarnings(
    as.integer(sub(".*\\[([0-9]+)\\].*", "\\1", df$param_name)))
  df$subj_idx[!grepl("\\[[0-9]+\\]$", df$param_name)] <- NA
  df
}

all_summaries <- lapply(all_models, load_summary)
names(all_summaries) <- all_models
combined <- bind_rows(all_summaries)

# Add log_kapp alongside kapp
log_kapp_rows <- combined %>%
  filter(param_base == "kapp") %>%
  mutate(
    across(c(mean, X50., sd, X2.5., X25., X75., X97.5.), log),
    param_base = "log_kapp",
    param_name = sub("kapp", "log_kapp", param_name)
  )
combined <- bind_rows(combined, log_kapp_rows)

# ── 3. Colour palette: one colour per model ──────────────────────────────────
model_colours <- c(
  "1-Joint-SVD-H" = "#E63946",
  "1-Joint-SVR-H" = "#F4A261",
  "2-Joint-SVD-H" = "#2A9D8F",
  "2-Joint-SVR-H" = "#457B9D",
  "3-Joint-SVD-H" = "#6A4C93",
  "3-Joint-SVR-H" = "#8AC926"
)

# ── 4. Theme ─────────────────────────────────────────────────────────────────
theme_param <- function() {
  theme_minimal(base_size = 11) +
    theme(
      strip.text         = element_text(face = "bold", size = 10),
      strip.background   = element_rect(fill = "#f0f0f0", color = NA),
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_blank(),
      plot.title         = element_text(face = "bold", size = 13),
      plot.subtitle      = element_text(size = 10, color = "grey40"),
      legend.position    = "bottom"
    )
}

# ── 5. One plot per model: subject posterior distributions ───────────────────
cat("\nGenerating per-model subject posterior plots...\n")

for (model_name in all_models) {
  
  colour <- model_colours[[model_name]]
  
  df <- combined %>%
    filter(model == model_name, param_type == "subject") %>%
    select(param_base, mean, X50.)
  
  if (nrow(df) == 0) {
    cat("No subject rows for", model_name, "— skipping\n")
    next
  }
  
  stats <- df %>%
    group_by(param_base) %>%
    summarise(
      grand_mean   = mean(mean, na.rm = TRUE),
      grand_median = mean(X50., na.rm = TRUE),
      .groups = "drop"
    )
  
  label_y <- df %>%
    group_by(param_base) %>%
    summarise(
      label_y = max(density(mean, na.rm = TRUE)$y) * 0.92,
      .groups = "drop"
    )
  
  stats <- stats %>% left_join(label_y, by = "param_base")
  
  p <- ggplot(df, aes(x = mean)) +
    geom_density(fill = colour, colour = colorspace::darken(colour, 0.3),
                 alpha = 0.35, linewidth = 0.7) +
    geom_rug(colour = colour, alpha = 0.4, linewidth = 0.3) +
    geom_vline(data = stats, aes(xintercept = grand_mean),
               colour = "grey20", linetype = "solid", linewidth = 0.8) +
    geom_text(data = stats,
              aes(x = grand_mean, y = label_y,
                  label = paste0("mean=", round(grand_mean, 3))),
              hjust = -0.1, vjust = 1, size = 3, colour = "grey20") +
    geom_vline(data = stats, aes(xintercept = grand_median),
               colour = "grey20", linetype = "dashed", linewidth = 0.8) +
    geom_text(data = stats,
              aes(x = grand_median, y = label_y * 0.78,
                  label = paste0("mdn=", round(grand_median, 3))),
              hjust = -0.1, vjust = 1, size = 3, colour = "grey20") +
    facet_wrap(~ param_base, scales = "free") +
    labs(
      title    = paste(model_name, "— Subject posterior distributions"),
      subtitle = "Solid line = mean  |  Dashed line = median",
      x        = "Posterior mean",
      y        = "Density"
    ) +
    theme_param()
  
  print(p)
  ggsave(paste0("figs/", model_name, "_subject_posteriors.pdf"),
         plot = p, width = 10, height = 6)
  cat("Saved:", model_name, "\n")
}

# ── 6. Noise parameter: RDM vs DD, per subject ───────────────────────────────
# The headline figure. Subject-level posterior means for the two noise
# parameters, on the log scale (where the copula is Gaussian), with the
# model-estimated correlation annotated.
cat("\nGenerating cross-task noise scatter plots...\n")

for (model_name in all_models) {
  
  info   <- get_model_info(model_name)
  colour <- model_colours[[model_name]]
  
  d <- combined %>%
    filter(model == model_name,
           param_base %in% c(info$noise_rdm, info$noise_dd),
           !is.na(subj_idx)) %>%
    select(param_base, subj_idx, mean) %>%
    pivot_wider(names_from = param_base, values_from = mean)
  
  if (nrow(d) == 0 || ncol(d) < 3) {
    cat("Noise parameters not found for", model_name, "— skipping\n"); next
  }
  
  names(d)[names(d) == info$noise_rdm] <- "noise_rdm"
  names(d)[names(d) == info$noise_dd]  <- "noise_dd"
  d <- d %>% mutate(log_rdm = log(noise_rdm), log_dd = log(noise_dd))
  
  # Model-estimated rho (latent, log scale) and the plug-in Pearson r
  smry <- all_summaries[[model_name]]
  rho_row <- smry[smry$param_name == info$rho_name, ]
  rho_txt <- if (nrow(rho_row) == 1) {
    sprintf("model rho = %.2f [%.2f, %.2f]",
            rho_row$mean, rho_row$X2.5., rho_row$X97.5.)
  } else "model rho unavailable"
  r_plugin <- cor(d$log_rdm, d$log_dd, use = "complete.obs")
  
  p <- ggplot(d, aes(x = log_rdm, y = log_dd)) +
    geom_point(colour = colour, alpha = 0.75, size = 2) +
    geom_smooth(method = "lm", formula = y ~ x, se = TRUE,
                colour = colorspace::darken(colour, 0.4),
                fill = colour, alpha = 0.15, linewidth = 0.7) +
    labs(
      title    = paste(model_name, "— Noise parameter across tasks"),
      subtitle = sprintf("%s  |  plug-in r on posterior means = %.2f",
                         rho_txt, r_plugin),
      x = paste0("log ", info$noise_rdm, "  (RDM)"),
      y = paste0("log ", info$noise_dd,  "  (DD)")
    ) +
    theme_param() +
    theme(legend.position = "none")
  
  print(p)
  ggsave(paste0("figs/", model_name, "_noise_scatter.pdf"),
         plot = p, width = 6.5, height = 5.5)
  cat("Saved scatter:", model_name, "\n")
}

# ── 7. Full latent correlation matrix, per model ─────────────────────────────
cat("\nGenerating correlation matrix heatmaps...\n")

for (model_name in all_models) {
  
  info <- get_model_info(model_name)
  smry <- all_summaries[[model_name]]
  
  om <- smry[grepl("^Omega\\[", smry$param_name), ]
  if (nrow(om) == 0) { cat("Omega absent for", model_name, "— skipping\n"); next }
  
  idx <- do.call(rbind, lapply(
    strsplit(gsub("Omega\\[|\\]", "", om$param_name), ","), as.numeric))
  
  d <- data.frame(
    row  = factor(info$dim_names[idx[, 1]], levels = info$dim_names),
    col  = factor(info$dim_names[idx[, 2]], levels = rev(info$dim_names)),
    corr = om$mean
  )
  
  p <- ggplot(d, aes(x = row, y = col, fill = corr)) +
    geom_tile(colour = "white", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.2f", corr)), size = 3.2, colour = "grey15") +
    scale_fill_gradient2(low = "#457B9D", mid = "white", high = "#E63946",
                         midpoint = 0, limits = c(-1, 1), name = expression(rho)) +
    coord_fixed() +
    labs(
      title    = paste(model_name, "— Latent correlation matrix"),
      subtitle = "Posterior means; all dimensions on the log scale except beta",
      x = "", y = ""
    ) +
    theme_param() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1),
          panel.grid  = element_blank())
  
  print(p)
  ggsave(paste0("figs/", model_name, "_Omega_heatmap.pdf"),
         plot = p, width = 6, height = 5.5)
  cat("Saved heatmap:", model_name, "\n")
}

# ── 8. Cross-model comparison of the noise correlation ───────────────────────
rho_df <- bind_rows(lapply(all_models, function(mn) {
  info <- get_model_info(mn)
  s <- all_summaries[[mn]]
  r <- s[s$param_name == info$rho_name, ]
  if (nrow(r) != 1) return(NULL)
  data.frame(model = mn, dataset = info$dataset,
             type = ifelse(info$is_svr, "SVR", "SVD"),
             mean = r$mean, lo = r$X2.5., hi = r$X97.5.)
}))

if (!is.null(rho_df) && nrow(rho_df) > 0) {
  rho_df$model <- factor(rho_df$model, levels = rev(all_models))
  p <- ggplot(rho_df, aes(x = mean, y = model, colour = model)) +
    geom_vline(xintercept = 0, linetype = 2, colour = "grey60") +
    geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.18, linewidth = 0.7) +
    geom_point(size = 2.8) +
    scale_colour_manual(values = model_colours, guide = "none") +
    coord_cartesian(xlim = c(-1, 1)) +
    labs(title = "Cross-task noise correlation across models",
         subtitle = "Posterior mean with 95% credible interval",
         x = expression(rho), y = "") +
    theme_param()
  print(p)
  ggsave("figs/joint_rho_comparison.pdf", p, width = 7, height = 4.5)
  write.csv(rho_df, "joint_rho_comparison.csv", row.names = FALSE)
}

cat("\n=== Done ===\n")
cat("PDFs saved in figs/\n")