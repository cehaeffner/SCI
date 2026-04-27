% AIC/BIC comparison: Separate fits (Softmax vs Luce)
clear;clc;

%% Load separate-fit data
rdm_softmax_1 = load('fitrdmdata_softmax_1.mat'); rdm_softmax_1 = rdm_softmax_1.rdm_1;
rdm_softmax_2 = load('fitrdmdata_softmax_2.mat'); rdm_softmax_2 = rdm_softmax_2.rdm_2;
rdm_softmax_3 = load('fitrdmdata_softmax_3.mat'); rdm_softmax_3 = rdm_softmax_3.rdm_3;
rdm_luce_1    = load('fitrdmdata_luce_1.mat');    rdm_luce_1    = rdm_luce_1.rdm_1;
rdm_luce_2    = load('fitrdmdata_luce_2.mat');    rdm_luce_2    = rdm_luce_2.rdm_2;
rdm_luce_3    = load('fitrdmdata_luce_3.mat');    rdm_luce_3    = rdm_luce_3.rdm_3;

dd_softmax_1  = load('fitdddata_softmax_1.mat');  dd_softmax_1  = dd_softmax_1.dd_1;
dd_softmax_2  = load('fitdddata_softmax_2.mat');  dd_softmax_2  = dd_softmax_2.dd_2;
dd_softmax_3  = load('fitdddata_softmax_3.mat');  dd_softmax_3  = dd_softmax_3.dd_3;
dd_luce_1     = load('fitdddata_luce_1.mat');     dd_luce_1     = dd_luce_1.dd_1;
dd_luce_2     = load('fitdddata_luce_2.mat');     dd_luce_2     = dd_luce_2.dd_2;
dd_luce_3     = load('fitdddata_luce_3.mat');     dd_luce_3     = dd_luce_3.dd_3;

%% Pack into cell arrays
cohort_labels = {'Cohort 1', 'Cohort 2', 'Cohort 3'};
rdm_softmax   = {rdm_softmax_1, rdm_softmax_2, rdm_softmax_3};
rdm_luce      = {rdm_luce_1,    rdm_luce_2,    rdm_luce_3};
dd_softmax    = {dd_softmax_1,  dd_softmax_2,  dd_softmax_3};
dd_luce       = {dd_luce_1,     dd_luce_2,     dd_luce_3};

k_sep = 2; % mu + alpha (RDM) or mu + kappa (DD)

%% Loop over cohorts
for c = 1:3
    rdm_sm = rdm_softmax{c};
    rdm_lc = rdm_luce{c};
    dd_sm  = dd_softmax{c};
    dd_lc  = dd_luce{c};

    rdm_sm_subs = [rdm_sm.subid]';
    rdm_lc_subs = [rdm_lc.subid]';
    dd_sm_subs  = [dd_sm.subid]';
    dd_lc_subs  = [dd_lc.subid]';
    common_subs = intersect(intersect(rdm_sm_subs, rdm_lc_subs), ...
                            intersect(dd_sm_subs,  dd_lc_subs));
    n = length(common_subs);

    aic_rdm_sm = nan(n,1); bic_rdm_sm = nan(n,1);
    aic_rdm_lc = nan(n,1); bic_rdm_lc = nan(n,1);
    aic_dd_sm  = nan(n,1); bic_dd_sm  = nan(n,1);
    aic_dd_lc  = nan(n,1); bic_dd_lc  = nan(n,1);

    for i = 1:n
        sid = common_subs(i);

        % RDM softmax
        s = rdm_sm(rdm_sm_subs == sid);
        if isfield(s, 'result_rdm') && ~isempty(s.result_rdm)
            nll           = s.result_rdm.modelLL;
            n_trials      = size(s.data, 1);
            aic_rdm_sm(i) = -2*nll + 2*k_sep;
            bic_rdm_sm(i) = -2*nll + k_sep*log(n_trials);
        end

        % RDM luce
        s = rdm_lc(rdm_lc_subs == sid);
        if isfield(s, 'result_rdm') && ~isempty(s.result_rdm)
            nll           = s.result_rdm.modelLL;
            n_trials      = size(s.data, 1);
            aic_rdm_lc(i) = -2*nll + 2*k_sep;
            bic_rdm_lc(i) = -2*nll + k_sep*log(n_trials);
        end

        % DD softmax
        s = dd_sm(dd_sm_subs == sid);
        if isfield(s, 'result_dd') && ~isempty(s.result_dd)
            nll          = s.result_dd.modelLL;
            n_trials     = size(s.data, 1);
            aic_dd_sm(i) = -2*nll + 2*k_sep;
            bic_dd_sm(i) = -2*nll + k_sep*log(n_trials);
        end

        % DD luce
        s = dd_lc(dd_lc_subs == sid);
        if isfield(s, 'result_dd') && ~isempty(s.result_dd)
            nll          = s.result_dd.modelLL;
            n_trials     = size(s.data, 1);
            aic_dd_lc(i) = -2*nll + 2*k_sep;
            bic_dd_lc(i) = -2*nll + k_sep*log(n_trials);
        end
    end

    % NaN filtering only
    valid_rdm = ~isnan(aic_rdm_sm) & ~isnan(aic_rdm_lc);
    valid_dd  = ~isnan(aic_dd_sm)  & ~isnan(aic_dd_lc);

    figure;
    sgtitle(sprintf('Separate Fits — %s', cohort_labels{c}));

    subplot(2,2,1);
    plot_bar_compare(aic_rdm_sm(valid_rdm), aic_rdm_lc(valid_rdm), 'AIC', 'RDM AIC'); axis square;

    subplot(2,2,2);
    plot_bar_compare(bic_rdm_sm(valid_rdm), bic_rdm_lc(valid_rdm), 'BIC', 'RDM BIC'); axis square;

    subplot(2,2,3);
    plot_bar_compare(aic_dd_sm(valid_dd), aic_dd_lc(valid_dd), 'AIC', 'DD AIC'); axis square;

    subplot(2,2,4);
    plot_bar_compare(bic_dd_sm(valid_dd), bic_dd_lc(valid_dd), 'BIC', 'DD BIC'); axis square;
end

%% Helper functions
function plot_bar_compare(vals_sm, vals_lc, ylbl, ttl)
    means = [mean(vals_sm), mean(vals_lc)];
    sems  = [std(vals_sm)/sqrt(length(vals_sm)), std(vals_lc)/sqrt(length(vals_lc))];
    b = bar(means, 'FaceColor', 'flat');
    b.CData = [0.6 0.4 0.8; 0.4 0.6 0.8];
    hold on;
    errorbar(1:2, means, sems, 'k', 'LineStyle', 'none', 'LineWidth', 1.2);
    [~,p] = ttest(vals_sm, vals_lc);
    y_top = max(means + sems);
    draw_bracket(1, 2, y_top + range(means)*0.1 - 1, stars(p));
    xticks(1:2); xticklabels({'Softmax','Luce'});
    ylabel(ylbl); title(ttl); box off;
end

function s = stars(p)
    if p < 0.001;     s = '***';
    elseif p < 0.01;  s = '**';
    elseif p < 0.05;  s = '*';
    else;             s = 'ns';
    end
end

function draw_bracket(x1, x2, y, label)
    tick = 0.005;
    plot([x1 x1 x2 x2], [y-tick y y y-tick], 'k', 'LineWidth', 1);
    text((x1+x2)/2, y + tick, label, 'HorizontalAlignment', 'center', 'FontSize', 11);
end