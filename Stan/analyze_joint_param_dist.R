library(ggplot2)
library(dplyr)
library(tidyr)
library(colorspace)

# ── 0. Model definitions ──────────────────────────────────────────────────────
joint_models <- c(
  "1-Joint-SVD-H", "1-Joint-SVR-H",
  "2-Joint-SVD-H", "2-Joint-SVR-H",
  "3-Joint-SVD-H", "3-Joint-SVR-H"
)
joint_models <- joint_models[file.exists(paste0(joint_models, "_summary.csv"))]
cat("Models found:", paste(joint_models, collapse = ", "), "\n\n")

# ── 1. Model metadata ─────────────────────────────────────────────────────────
get_model_info <- function(model_name) {
  is_svr  <- grepl("SVR", model_name)
  dataset <- sub("-Joint.*", "", model_name)
  is_ds3  <- dataset == "3"
  noise_subj <- ifelse(is_svr, "mu", "gamma")
  noise_par  <- ifelse(is_svr, "mm", "gm")
  noise_sd   <- ifelse(is_svr, "ms", "gs")
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
    hyper_pars = hyper_pars,
    subj_pars  = subj_pars
  )
}

# ── 2. Load summaries ─────────────────────────────────────────────────────────
load_summary <- function(model_name) {
  info <- get_model_info(model_name)
  df   <- read.csv(paste0(model_name, "_summary.csv"), row.names = 1)
  df$param_name <- rownames(df)
  df$model      <- model_name
  df$is_svr     <- info$is_svr
  df$dataset    <- info$dataset

  subj_pat  <- paste0("^(", paste(info$subj_pars,  collapse = "|"), ")\\[")
  hyper_pat <- paste0("^(", paste(info$hyper_pars, collapse = "|"), ")$")
  df$param_type <- dplyr::case_when(
    grepl(subj_pat,  df$param_name) ~ "subject",
    grepl(hyper_pat, df$param_name) ~ "hyper",
    TRUE                            ~ "other"
  )
  df$param_base <- sub("\\[.*", "", df$param_name)
  df
}

all_summaries <- lapply(joint_models, load_summary)
combined      <- dplyr::bind_rows(all_summaries)

# Add log_kapp alongside kapp
log_kapp_rows <- combined %>%
  filter(param_base == "kapp") %>%
  mutate(
    across(c(mean, X50., sd, X2.5., X25., X75., X97.5.), log),
    param_base = "log_kapp",
    param_name = sub("kapp", "log_kapp", param_name)
  )
combined <- dplyr::bind_rows(combined, log_kapp_rows)

# ── 3. Colour palette ─────────────────────────────────────────────────────────
model_colours <- c(
  "1-Joint-SVD-H" = "#E63946",
  "1-Joint-SVR-H" = "#F4A261",
  "2-Joint-SVD-H" = "#2A9D8F",
  "2-Joint-SVR-H" = "#457B9D",
  "3-Joint-SVD-H" = "#9B2226",
  "3-Joint-SVR-H" = "#6A4C93"
)

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
cat("\nGenerating per-model subject posterior plots...\n")

for (model_name in joint_models) {

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
      grand_mean   = mean(mean,  na.rm = TRUE),
      grand_median = mean(X50.,  na.rm = TRUE),
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
      subtitle = "Solid = mean  |  Dashed = median",
      x        = "Posterior mean",
      y        = "Density"
    ) +
    theme_param()

  print(p)
  ggsave(paste0("figs/", model_name, "_subject_posteriors.pdf"),
         plot = p, width = 10, height = 6)
  cat("Saved:", model_name, "\n")
}

# ── 6. Cross-model comparison: SVD vs SVR per dataset ────────────────────────
cat("\nGenerating SVD vs SVR comparison plots...\n")

for (ds in c("1", "2", "3")) {

  svd_name <- paste0(ds, "-Joint-SVD-H")
  svr_name <- paste0(ds, "-Joint-SVR-H")
  available <- intersect(c(svd_name, svr_name), joint_models)
  if (length(available) < 2) next

  df_cmp <- combined %>%
    filter(model %in% available, param_type == "subject") %>%
    mutate(model_type = ifelse(grepl("SVD", model), "SVD", "SVR"))

  # Shared parameters: alph, kapp (and beta if ds3)
  shared_pars <- c("alph", "log_kapp")
  if (ds == "3") shared_pars <- c(shared_pars, "beta")

  df_cmp <- df_cmp %>% filter(param_base %in% shared_pars)
  if (nrow(df_cmp) == 0) next

  p <- ggplot(df_cmp, aes(x = mean, fill = model_type, colour = model_type)) +
    geom_density(alpha = 0.35, linewidth = 0.7) +
    facet_wrap(~ param_base, scales = "free") +
    scale_fill_manual(values  = c("SVD" = "#E63946", "SVR" = "#457B9D")) +
    scale_colour_manual(values = c("SVD" = "#C1121F", "SVR" = "#1D3557")) +
    labs(
      title    = paste0("Dataset ", ds, " — SVD vs SVR: Shared parameter distributions"),
      subtitle = "alph = risk preference  |  log_kapp = log discounting rate",
      x        = "Posterior mean",
      y        = "Density",
      fill     = "Model type",
      colour   = "Model type"
    ) +
    theme_param()

  print(p)
  ggsave(paste0("figs/", ds, "_joint_svd_vs_svr_params.pdf"),
         plot = p, width = 10, height = 6)
  cat("SVD vs SVR comparison saved for dataset", ds, "\n")
}

# ── 7. Cross-dataset comparison: shared parameters by dataset ────────────────
cat("\nGenerating cross-dataset parameter plots...\n")

for (mtype in c("SVD", "SVR")) {

  df_ds <- combined %>%
    filter(grepl(mtype, model), param_type == "subject",
           param_base %in% c("alph", "log_kapp")) %>%
    mutate(dataset = paste0("Dataset ", dataset))

  if (nrow(df_ds) == 0) next

  p <- ggplot(df_ds, aes(x = mean, fill = dataset, colour = dataset)) +
    geom_density(alpha = 0.30, linewidth = 0.7) +
    facet_wrap(~ param_base, scales = "free") +
    labs(
      title    = paste0("Joint-", mtype, " — Parameter distributions across datasets"),
      x        = "Posterior mean",
      y        = "Density",
      fill     = "Dataset",
      colour   = "Dataset"
    ) +
    theme_param()

  print(p)
  ggsave(paste0("figs/joint_", mtype, "_cross_dataset_params.pdf"),
         plot = p, width = 10, height = 5)
  cat("Cross-dataset plot saved for", mtype, "\n")
}

cat("\n=== Done ===\n")
cat("PDFs saved in figs/\n")
