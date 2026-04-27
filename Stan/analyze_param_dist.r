library(ggplot2)
library(dplyr)
library(tidyr)
library(colorspace)

# ── 0. Model definitions (mirror analyze_models.R) ──────────────────────────
dd_models  <- c("1-DD-SVD-H",  "1-DD-SVR-H",
                "2-DD-SVD-H",  "2-DD-SVR-H",
                "3-DD-SVD-H",  "3-DD-SVR-H")

rdm_models <- c("1-RDM-SVD-H", "1-RDM-SVR-H",
                "2-RDM-SVD-H", "2-RDM-SVR-H",
                "3-RDM-SVD-H", "3-RDM-SVR-H")

all_models <- c(dd_models, rdm_models)
all_models <- all_models[file.exists(paste0(all_models, "_summary.csv"))]
cat("Models found:", paste(all_models, collapse = ", "), "\n\n")

# ── 1. Model metadata ────────────────────────────────────────────────────────
get_model_info <- function(model_name) {
  is_svr <- grepl("SVR", model_name)
  is_rdm <- grepl("RDM", model_name)
  ds     <- sub("-(DD|RDM).*", "", model_name)
  list(
    name       = model_name,
    is_svr     = is_svr,
    is_rdm     = is_rdm,
    dataset    = ds,
    hyper_pars = if      (is_rdm & !is_svr)  c("am", "gm", "as", "gs")
    else if (is_rdm &  is_svr)  c("am", "mm", "as", "ms")
    else if (!is_rdm & !is_svr) c("km", "gm", "ks", "gs")
    else                        c("km", "mm", "ks", "ms"),
    subj_pars  = {
      base <- if      (is_rdm &  is_svr)  c("alph", "mu")
      else if (is_rdm & !is_svr)  c("alph", "gamma")
      else if (!is_rdm & !is_svr) c("kapp", "gamma")
      else                        c("kapp", "mu")       # DD SVR
      if (is_rdm & ds == "3") c(base, "beta") else base
    }
  )
}

# ── 2. Load summaries ────────────────────────────────────────────────────────
load_summary <- function(model_name) {
  info <- get_model_info(model_name)
  df   <- read.csv(paste0(model_name, "_summary.csv"), row.names = 1)
  df$param_name  <- rownames(df)
  df$model       <- model_name
  df$is_svr      <- info$is_svr
  df$is_rdm      <- info$is_rdm
  df$dataset     <- sub("-(DD|RDM).*", "", model_name)
  
  subj_pat  <- paste0("^(", paste(info$subj_pars,  collapse = "|"), ")\\[")
  hyper_pat <- paste0("^(", paste(info$hyper_pars, collapse = "|"), ")$")
  df$param_type <- case_when(
    grepl(subj_pat,  df$param_name) ~ "subject",
    grepl(hyper_pat, df$param_name) ~ "hyper",
    TRUE                            ~ "other"
  )
  
  df$param_base <- sub("\\[.*", "", df$param_name)
  df
}

all_summaries <- lapply(all_models, load_summary)
combined      <- bind_rows(all_summaries)

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
# Distinct palette — one colour per model name
model_colours <- c(
  "1-DD-SVD-H"  = "#E63946",
  "1-DD-SVR-H"  = "#F4A261",
  "2-DD-SVD-H"  = "#2A9D8F",
  "2-DD-SVR-H"  = "#457B9D",
  "3-DD-SVD-H"  = "#9B2226",
  "3-DD-SVR-H"  = "#AE2012",
  "1-RDM-SVD-H" = "#6A4C93",
  "1-RDM-SVR-H" = "#1982C4",
  "2-RDM-SVD-H" = "#8AC926",
  "2-RDM-SVR-H" = "#FF595E",
  "3-RDM-SVD-H" = "#FFCA3A",
  "3-RDM-SVR-H" = "#6A994E"
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
  
  # Per-parameter grand mean and median
  stats <- df %>%
    group_by(param_base) %>%
    summarise(
      grand_mean   = mean(mean, na.rm = TRUE),
      grand_median = mean(X50., na.rm = TRUE),
      .groups = "drop"
    )
  
  # Y position for labels: place near top of each panel by computing
  # per-parameter density peak as a proxy (use a high quantile of mean values)
  label_y <- df %>%
    group_by(param_base) %>%
    summarise(
      label_y = max(density(mean, na.rm = TRUE)$y) * 0.92,
      .groups = "drop"
    )
  
  stats <- stats %>% left_join(label_y, by = "param_base")
  
  p <- ggplot(df, aes(x = mean)) +
    geom_density(fill = colour, colour = colorspace::darken(colour, 0.3), alpha = 0.35, linewidth = 0.7) +
    geom_rug(colour = colour, alpha = 0.4, linewidth = 0.3) +
    # Mean line + label
    geom_vline(
      data = stats, aes(xintercept = grand_mean),
      colour = "grey20", linetype = "solid", linewidth = 0.8
    ) +
    geom_text(
      data = stats,
      aes(x = grand_mean, y = label_y,
          label = paste0("mean=", round(grand_mean, 3))),
      hjust = -0.1, vjust = 1, size = 3, colour = "grey20"
    ) +
    # Median line + label
    geom_vline(
      data = stats, aes(xintercept = grand_median),
      colour = "grey20", linetype = "dashed", linewidth = 0.8
    ) +
    geom_text(
      data = stats,
      aes(x = grand_median, y = label_y * 0.78,
          label = paste0("mdn=", round(grand_median, 3))),
      hjust = -0.1, vjust = 1, size = 3, colour = "grey20"
    ) +
    facet_wrap(~ param_base, scales = "free") +
    labs(
      title    = paste(model_name, "— Subject posterior distributions"),
      subtitle = "Solid line = mean  |  Dashed line = median",
      x        = "Posterior mean",
      y        = "Density"
    ) +
    theme_param()
  
  print(p)
  ggsave(
    paste0("figs/", model_name, "_subject_posteriors.pdf"),
    plot = p, width = 10, height = 6
  )
  cat("Saved:", model_name, "\n")
}

cat("\n=== Done ===\n")
cat("PDFs saved in figs/\n")