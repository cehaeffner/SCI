% AIC/BIC comparison: Joint fits (Softmax vs Luce)
clear;clc;

%% Load joint-fit data
joint_softmax_1 = load('fitjointdata_softmax_1.mat'); joint_softmax_1 = joint_softmax_1.rdm_1;
joint_softmax_2 = load('fitjointdata_softmax_2.mat'); joint_softmax_2 = joint_softmax_2.rdm_2;
joint_softmax_3 = load('fitjointdata_softmax_3.mat'); joint_softmax_3 = joint_softmax_3.rdm_3;
joint_luce_1    = load('fitjointdata_luce_1.mat');    joint_luce_1    = joint_luce_1.rdm_1;
joint_luce_2    = load('fitjointdata_luce_2.mat');    joint_luce_2    = joint_luce_2.rdm_2;
joint_luce_3    = load('fitjointdata_luce_3.mat');    joint_luce_3    = joint_luce_3.rdm_3;

% Load DD data to get n_trials for joint (RDM + DD combined)
dd_softmax_1 = load('dd_softmax_1.mat'); dd_softmax_1 = dd_softmax_1.alldata;
dd_softmax_2 = load('dd_softmax_2.mat'); dd_softmax_2 = dd_softmax_2.alldata;
dd_softmax_3 = load('dd_softmax_3.mat'); dd_softmax_3 = dd_softmax_3.alldata;
dd_luce_1    = load('dd_luce_1.mat');    dd_luce_1    = dd_luce_1.alldata;
dd_luce_2    = load('dd_luce_2.mat');    dd_luce_2    = dd_luce_2.alldata;
dd_luce_3    = load('dd_luce_3.mat');    dd_luce_3    = dd_luce_3.alldata;

%% Pack into cell arrays
cohort_labels  = {'Cohort 1', 'Cohort 2', 'Cohort 3'};
joint_softmax  = {joint_softmax_1, joint_softmax_2, joint_softmax_3};
joint_luce     = {joint_luce_1,    joint_luce_2,    joint_luce_3};
dd_softmax_all = {dd_softmax_1,    dd_softmax_2,    dd_softmax_3};
dd_luce_all    = {dd_luce_1,       dd_luce_2,       dd_luce_3};

k_joint = 3; % mu, alpha, kappa

%% Loop over cohorts
for c = 1:3
    jt_sm  = joint_softmax{c};
    jt_lc  = joint_luce{c};
    dd_sm  = dd_softmax_all{c};
    dd_lc  = dd_luce_all{c};

    jt_sm_subs = [jt_sm.subid]';
    jt_lc_subs = [jt_lc.subid]';
    dd_sm_subs = [dd_sm.subid]';
    dd_lc_subs = [dd_lc.subid]';
    common_subs = intersect(jt_sm_subs, jt_lc_subs);
    n = length(common_subs);

    aic_sm = nan(n,1); bic_sm = nan(n,1);
    aic_lc = nan(n,1); bic_lc = nan(n,1);

    for i = 1:n
        sid = common_subs(i);

        % Joint softmax
        s    = jt_sm(jt_sm_subs == sid);
        dd_s = dd_sm(dd_sm_subs == sid);
        if isfield(s, 'result_joint') && ~isempty(s.result_joint)
            nll      = s.result_joint.modelLL;
            n_trials = size(s.data, 1) + size(dd_s.data, 1);
            aic_sm(i) = -2*nll + 2*k_joint;
            bic_sm(i) = -2*nll + k_joint*log(n_trials);
        end

        % Joint luce
        s    = jt_lc(jt_lc_subs == sid);
        dd_s = dd_lc(dd_lc_subs == sid);
        if isfield(s, 'result_joint') && ~isempty(s.result_joint)
            nll      = s.result_joint.modelLL;
            n_trials = size(s.data, 1) + size(dd_s.data, 1);
            aic_lc(i) = -2*nll + 2*k_joint;
            bic_lc(i) = -2*nll + k_joint*log(n_trials);
        end
    end

    % NaN filtering
    valid = ~isnan(aic_sm) & ~isnan(aic_lc);

    figure;
    sgtitle(sprintf('Joint Fits — %s', cohort_labels{c}));

    subplot(1,2,1);
    plot_bar_compare(aic_sm(valid), aic_lc(valid), 'AIC', 'Joint AIC'); axis square;

    subplot(1,2,2);
    plot_bar_compare(bic_sm(valid), bic_lc(valid), 'BIC', 'Joint BIC'); axis square;
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
    draw_bracket(1, 2, y_top + range(means)*0.1 + 0.5, stars(p));
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