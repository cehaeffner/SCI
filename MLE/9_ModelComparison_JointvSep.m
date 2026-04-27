% Compare separate vs joint model fit (pseudo-R2)
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

%% Load joint-fit data
joint_softmax_1 = load('fitjointdata_softmax_1.mat'); joint_softmax_1 = joint_softmax_1.rdm_1;
joint_softmax_2 = load('fitjointdata_softmax_2.mat'); joint_softmax_2 = joint_softmax_2.rdm_2;
joint_softmax_3 = load('fitjointdata_softmax_3.mat'); joint_softmax_3 = joint_softmax_3.rdm_3;
joint_luce_1    = load('fitjointdata_luce_1.mat');    joint_luce_1    = joint_luce_1.rdm_1;
joint_luce_2    = load('fitjointdata_luce_2.mat');    joint_luce_2    = joint_luce_2.rdm_2;
joint_luce_3    = load('fitjointdata_luce_3.mat');    joint_luce_3    = joint_luce_3.rdm_3;

%% Pack into cell arrays for looping
cohort_labels = {'Cohort 1', 'Cohort 2', 'Cohort 3'};
model_labels  = {'Softmax', 'Luce'};

rdm_sep   = {{rdm_softmax_1, rdm_softmax_2, rdm_softmax_3}, ...
             {rdm_luce_1,    rdm_luce_2,    rdm_luce_3}};
dd_sep    = {{dd_softmax_1,  dd_softmax_2,  dd_softmax_3}, ...
             {dd_luce_1,     dd_luce_2,     dd_luce_3}};
joint_fit = {{joint_softmax_1, joint_softmax_2, joint_softmax_3}, ...
             {joint_luce_1,    joint_luce_2,    joint_luce_3}};

%% Loop over cohorts — one figure per cohort
for c = 1:3
    figure;
    sgtitle(cohort_labels{c});

    for m = 1:2
        rdm_data   = rdm_sep{m}{c};
        dd_data    = dd_sep{m}{c};
        joint_data = joint_fit{m}{c};

        % --- Match subjects across all three structs ---
        rdm_subids   = [rdm_data.subid]';
        dd_subids    = [dd_data.subid]';
        joint_subids = [joint_data.subid]';
        common_subs  = intersect(intersect(rdm_subids, dd_subids), joint_subids);
        n            = length(common_subs);

        pr2_rdm   = nan(n, 1);
        pr2_dd    = nan(n, 1);
        pr2_joint = nan(n, 1);

        for i = 1:n
            sid = common_subs(i);
            pr2_rdm(i)   = rdm_data(rdm_subids     == sid).pr2_rdm;
            pr2_dd(i)    = dd_data(dd_subids        == sid).pr2_dd;
            pr2_joint(i) = joint_data(joint_subids  == sid).pr2_joint;
        end

        % --- NaN filtering (must come after vectors are populated) ---
        valid_rdm_joint = ~isnan(pr2_rdm) & ~isnan(pr2_joint);
        valid_dd_joint  = ~isnan(pr2_dd)  & ~isnan(pr2_joint);
        valid_all       = ~isnan(pr2_rdm) & ~isnan(pr2_dd) & ~isnan(pr2_joint);

        pr2_rdm_c   = pr2_rdm(valid_all);
        pr2_dd_c    = pr2_dd(valid_all);
        pr2_joint_c = pr2_joint(valid_all);
        n_valid     = sum(valid_all);

        % --- Scatter: RDM separate vs Joint (equal axes) ---
        subplot(2, 3, (m-1)*3 + 1);
        x = pr2_rdm(valid_rdm_joint); y = pr2_joint(valid_rdm_joint);
        ax_min = min([x; y]) - 0.05; ax_max = max([x; y]) + 0.05;
        scatter(x, y, 40, 'filled'); hold on;
        plot([ax_min ax_max], [ax_min ax_max], 'r-', 'LineWidth', 1.2);
        xlim([ax_min ax_max]); ylim([ax_min ax_max]);
        xlabel('R^2 (RDM separate)'); ylabel('R^2 (Joint)');
        [r,p] = corr(x, y);
        title(sprintf('%s: RDM sep vs Joint | r=%.2f, p=%.3f', model_labels{m}, r, p));
        box off; axis square;

        % --- Scatter: DD separate vs Joint (equal axes) ---
        subplot(2, 3, (m-1)*3 + 2);
        x = pr2_dd(valid_dd_joint); y = pr2_joint(valid_dd_joint);
        ax_min = min([x; y]) - 0.05; ax_max = max([x; y]) + 0.05;
        scatter(x, y, 40, 'filled'); hold on;
        plot([ax_min ax_max], [ax_min ax_max], 'r-', 'LineWidth', 1.2);
        xlim([ax_min ax_max]); ylim([ax_min ax_max]);
        xlabel('R^2 (DD separate)'); ylabel('R^2 (Joint)');
        [r,p] = corr(x, y);
        title(sprintf('%s: DD sep vs Joint | r=%.2f, p=%.3f', model_labels{m}, r, p));
        box off; axis square;

        % --- Repeated measures ANOVA on NaN-filtered data ---
        rm_table  = table(pr2_rdm_c, pr2_dd_c, pr2_joint_c, ...
                          'VariableNames', {'RDM','DD','Joint'});
        within    = table({'RDM';'DD';'Joint'}, 'VariableNames', {'Model'});
        rm        = fitrm(rm_table, 'RDM,DD,Joint~1', 'WithinDesign', within);
        ranova_result = ranova(rm);

        % Bonferroni-corrected post-hoc paired t-tests (3 comparisons)
        [~, p_rdm_dd]    = ttest(pr2_rdm_c, pr2_dd_c);
        [~, p_rdm_joint] = ttest(pr2_rdm_c, pr2_joint_c);
        [~, p_dd_joint]  = ttest(pr2_dd_c,  pr2_joint_c);
        p_rdm_dd    = min(p_rdm_dd    * 3, 1);
        p_rdm_joint = min(p_rdm_joint * 3, 1);
        p_dd_joint  = min(p_dd_joint  * 3, 1);

        % --- Bar plot ---
        subplot(2, 3, (m-1)*3 + 3);
        means = [mean(pr2_rdm_c), mean(pr2_dd_c), mean(pr2_joint_c)];
        sems  = [std(pr2_rdm_c)/sqrt(n_valid), ...
                 std(pr2_dd_c)/sqrt(n_valid), ...
                 std(pr2_joint_c)/sqrt(n_valid)];
        b = bar(means, 'FaceColor', 'flat');
        b.CData = [0.6 0.4 0.8; 0.4 0.6 0.8; 0.4 0.8 0.6];
        hold on;
        errorbar(1:3, means, sems, 'k', 'LineStyle', 'none', 'LineWidth', 1.2);
        xticks(1:3); xticklabels({'RDM', 'DD', 'Joint'});
        ylabel('Mean pseudo-R^2');
        anova_p = ranova_result.pValue(1);
        title(sprintf('%s: Mean R^2 by model (RM ANOVA p=%.3f)', model_labels{m}, anova_p));
        box off;

        % Draw significance brackets
        y_top = max(means + sems);
        gap   = 0.02;
        draw_bracket(1, 2, y_top + gap*1, stars(p_rdm_dd));
        draw_bracket(2, 3, y_top + gap*2, stars(p_dd_joint));
        draw_bracket(1, 3, y_top + gap*3, stars(p_rdm_joint));
    end
end

%% Helper functions
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