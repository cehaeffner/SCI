library(ggplot2)
library(dplyr)
library(tidyr)
library(colorspace)

dir.create("figs", showWarnings = FALSE)

# ── 0. Define joint models ────────────────────────────────────────────────────
joint_models <- c(
  "1-DD-SVD-copula-RDM-SVD",
  "1-DD-SVR-copula-RDM-SVR",
  "1-DD-SVD-copula-RDM-amb-SVD",
  "1-DD-SVR-copula-RDM-amb-SVR"
  # add further datasets here
)

joint_models <- joint_models[file.exists(paste0(joint_models, "_summary.csv"))]
cat("Models found:", paste(joint_models, collapse = ", "), "\n\n")

# ── 1. Model metadata ─────────────────────────────────────────────────────────
# All parameters are fully task-specific. Nothing is shared between tasks.
# The copula only links latent raw scores; rho is the sole cross-task quantity.
get_model_info <- function(model_name) {
  is_svr  <- grepl("SVR", model_name)
  is_amb  <- grepl("amb", model_name)
  dataset <- sub("-(DD).*", "", model_name)
  list(
    name    = model_name,
    is_svr  = is_svr,
    is_amb  = is_amb,
    dataset = dataset,
    rdm_hyper = c(
      "am", "as",
      if (!is_svr) c("gm_rdm", "gs_rdm") else c("mm_rdm", "ms_rdm"),
      if (is_amb)  c("bm", "bs")          else NULL
    ),
    dd_hyper = c(
      "km", "ks",
      if (!is_svr) c("gm_dd", "gs_dd") else c("mm_dd", "ms_dd")
    ),
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

# ── 2. Load summaries ─────────────────────────────────────────────────────────
load_summary <- function(model_name) {
  info <- get_model_info(model_name)
  df   <- read.csv(paste0(model_name, "_summary.csv"), row.names = 1)
  df$param_name <- rownames(df)
  df$model      <- model_name
  df$is_svr     <- info$is_svr
  df$is_amb     <- info$is_amb
  df$dataset    <- info$dataset

  all_subj  <- c(info$rdm_subj, info$dd_subj)
  all_hyper <- c(info$rdm_hyper, info$dd_hyper)
  subj_pat  <- paste0("^(", paste(all_subj,  collapse = "|"), ")\\[")
  hyper_pat <- paste0("^(", paste(all_hyper, collapse = "|"), ")$")

  df$param_type <- case_when(
    grepl(subj_pat,  df$param_name) ~ "subject",
    grepl(hyper_pat, df$param_name) ~ "hyper",
    df$param_name == "rho_out"      ~ "rho",
    TRUE                            ~ "other"
  )
  df$param_base <- sub("\\[.*", "", df$param_name)

  # Tag each row with which task it belongs to
  df$task <- case_when(
    df$param_base %in% info$rdm_subj  ~ "RDM",
    df$param_base %in% info$rdm_hyper ~ "RDM",
    df$param_base %in% info$dd_subj   ~ "DD",
    df$param_base %in% info$dd_hyper  ~ "DD",
    df$param_name == "rho_out"        ~ "copula",
    TRUE                               ~ NA_character_
  )
  df
}

all_summaries <- lapply(joint_models, load_summary)
combined      <- bind_rows(all_summaries)

# Add log_kapp (DD discounting on log scale)
log_kapp_rows <- combined %>%
  filter(param_base == "kapp") %>%
  mutate(
    across(c(mean, X50., sd, X2.5., X25., X75., X97.5.), log),
    param_base = "log_kapp",
    param_name = sub("kapp", "log_kapp", param_name)
  )
combined <- bind_rows(combined, log_kapp_rows)

# ── 3. Colour palette ─────────────────────────────────────────────────────────
model_colours <- c(
  "1-DD-SVD-copula-RDM-SVD"     = "#E63946",
  "1-DD-SVR-copula-RDM-SVR"     = "#457B9D",
  "1-DD-SVD-copula-RDM-amb-SVD" = "#2A9D8F",
  "1-DD-SVR-copula-RDM-amb-SVR" = "#9B2226"
)

task_colours <- c("RDM" = "#1d3557", "DD" = "#e76f51")

# ── 4. Theme ──────────────────────────────────────────────────────────────────
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

# ── 5. Per-model subject posterior distributions ──────────────────────────────
# Panels are coloured by task (RDM vs DD) to make clear these are independent
# estimates — they happen to be in the same plot only for easy comparison.
cat("\nGenerating per-model subject posterior plots...\n")

for (model_name in joint_models) {

  df <- combined %>%
    filter(model == model_name, param_type == "subject") %>%
    select(param_base, task, mean, X50.)

  if (nrow(df) == 0) { cat("No subject rows for", model_name, "— skipping\n"); next }

  stats <- df %>%
    group_by(param_base, task) %>%
    summarise(grand_mean = mean(mean, na.rm = TRUE),
              grand_median = mean(X50., na.rm = TRUE),
              .groups = "drop")

  label_y <- df %>%
    group_by(param_base) %>%
    summarise(label_y = max(density(mean, na.rm = TRUE)$y) * 0.92, .groups = "drop")

  stats <- stats %>% left_join(label_y, by = "param_base")

  p <- ggplot(df, aes(x = mean, fill = task, colour = task)) +
    geom_density(alpha = 0.30, linewidth = 0.7) +
    geom_rug(alpha = 0.3, linewidth = 0.3) +
    geom_vline(data = stats, aes(xintercept = grand_mean, colour = task),
               linetype = "solid", linewidth = 0.8) +
    geom_vline(data = stats, aes(xintercept = grand_median, colour = task),
               linetype = "dashed", linewidth = 0.8) +
    geom_text(data = stats,
              aes(x = grand_mean, y = label_y,
                  label = paste0("M=", round(grand_mean, 3)), colour = task),
              hjust = -0.1, vjust = 1, size = 2.8, show.legend = FALSE) +
    scale_fill_manual(values = task_colours) +
    scale_colour_manual(values = task_colours) +
    facet_wrap(~ param_base, scales = "free") +
    labs(
      title    = paste(model_name, "— Subject posterior distributions"),
      subtitle = "RDM and DD parameters are estimated independently. Colour = task.",
      x = "Posterior mean", y = "Density", fill = "Task", colour = "Task"
    ) +
    theme_param()

  print(p)
  ggsave(paste0("figs/", model_name, "_subject_posteriors.pdf"),
         plot = p, width = 12, height = 7)
  cat("Saved:", model_name, "\n")
}

# ── 6. Rho posterior summary across models ────────────────────────────────────
cat("\nGenerating rho summary plot...\n")

rho_summary <- combined %>%
  filter(param_base == "rho_out") %>%
  select(model, mean, X2.5., X50., X97.5.) %>%
  mutate(
    model = factor(model, levels = joint_models),
    sig   = ifelse(X2.5. > 0 | X97.5. < 0, "*", "")
  )

if (nrow(rho_summary) > 0) {
  p_rho <- ggplot(rho_summary, aes(x = model, y = mean, colour = model)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60") +
    geom_pointrange(aes(ymin = X2.5., ymax = X97.5.), size = 0.8, linewidth = 1.2) +
    geom_text(aes(label = sig, y = X97.5. + 0.03),
              size = 5, colour = "black", vjust = 0) +
    scale_colour_manual(values = model_colours[levels(rho_summary$model)]) +
    labs(
      title    = "Copula correlation (rho) across joint models",
      subtitle = "Point = posterior mean, range = 95% CI. * = CI excludes zero.",
      x = "", y = "rho"
    ) +
    theme_param() +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 20, hjust = 1))

  print(p_rho)
  ggsave("figs/rho_summary.pdf", plot = p_rho, width = 8, height = 5)
  cat("Rho summary plot saved.\n")
}

# ── 7. Cross-task scatter: DD vs RDM choice-rule parameter per subject ─────────
# This plots the independently estimated parameters against each other.
# The regression slope reflects the empirical correlation; rho from the model
# is the Bayesian estimate of that same correlation.
cat("\nGenerating cross-task scatter plots...\n")

for (model_name in joint_models) {

  info   <- get_model_info(model_name)
  colour <- model_colours[[model_name]]

  cr_rdm   <- if (!info$is_svr) "gamma_rdm" else "mu_rdm"
  cr_dd    <- if (!info$is_svr) "gamma_dd"  else "mu_dd"
  cr_label <- if (!info$is_svr) "gamma (sensitivity)" else "mu (noise)"

  df_rdm <- combined %>%
    filter(model == model_name, param_base == cr_rdm, param_type == "subject") %>%
    mutate(subj = as.integer(sub(".*\\[(.*)\\]", "\\1", param_name))) %>%
    select(subj, rdm_val = mean)

  df_dd <- combined %>%
    filter(model == model_name, param_base == cr_dd, param_type == "subject") %>%
    mutate(subj = as.integer(sub(".*\\[(.*)\\]", "\\1", param_name))) %>%
    select(subj, dd_val = mean)

  df_scatter <- inner_join(df_rdm, df_dd, by = "subj")
  if (nrow(df_scatter) == 0) next

  rho_val <- combined %>%
    filter(model == model_name, param_base == "rho_out") %>%
    pull(mean)
  rho_lab <- if (length(rho_val) > 0) sprintf("rho = %.3f", rho_val[1]) else ""

  p_scatter <- ggplot(df_scatter, aes(x = rdm_val, y = dd_val)) +
    geom_point(colour = colour, alpha = 0.7, size = 2) +
    geom_smooth(method = "lm", se = TRUE, colour = "grey30", linewidth = 0.8) +
    annotate("text", x = Inf, y = Inf, label = rho_lab,
             hjust = 1.1, vjust = 1.5, size = 4, colour = "grey30") +
    labs(
      title    = paste(model_name, "— Cross-task", cr_label),
      subtitle = "Each point = one subject's posterior mean. Parameters estimated independently per task.",
      x = paste("RDM", cr_label),
      y = paste("DD",  cr_label)
    ) +
    theme_param()

  print(p_scatter)
  ggsave(paste0("figs/", model_name, "_cross_task_scatter.pdf"),
         plot = p_scatter, width = 6, height = 5)
  cat("Scatter saved:", model_name, "\n")
}

cat("\n=== Done ===\n")
cat("PDFs saved in figs/\n")
