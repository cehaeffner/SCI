%% Plot parameter distributions for all joint model fits
clear; clc; close all;

param_names = {'mu', 'alpha', 'kappa'};
n_params    = length(param_names);

files = {
    'fitjointdata_softmax_1.mat', 'rdm_1', 'Cohort 1 Softmax';
    'fitjointdata_luce_1.mat',    'rdm_1', 'Cohort 1 Luce';
    'fitjointdata_softmax_2.mat', 'rdm_2', 'Cohort 2 Softmax';
    'fitjointdata_luce_2.mat',    'rdm_2', 'Cohort 2 Luce';
    'fitjointdata_softmax_3.mat', 'rdm_3', 'Cohort 3 Softmax';
    'fitjointdata_luce_3.mat',    'rdm_3', 'Cohort 3 Luce';
};

n_files = size(files, 1);

%% Extract parameters
all_params = cell(n_files, 1);  % each cell: n_subjects x 3

for f = 1:n_files
    if ~isfile(files{f,1})
        warning('File not found: %s', files{f,1});
        all_params{f} = [];
        continue
    end
    tmp  = load(files{f,1});
    data = tmp.(files{f,2});

    b_mat = [];
    for s = 1:length(data)
        if isfield(data(s), 'b_joint') && ~isempty(data(s).b_joint)
            b_mat = [b_mat; data(s).b_joint(:)']; %#ok<AGROW>
        end
    end
    all_params{f} = b_mat;
end

%% Plot: one figure per parameter, all 6 models overlaid
colors = lines(n_files);

for p = 1:n_params
    figure('Name', param_names{p}, 'Position', [100 100 900 500]);

    for f = 1:n_files
        if isempty(all_params{f}), continue; end
        vals = all_params{f}(:, p);

        subplot(2, 3, f);
        histogram(vals, 15, 'FaceColor', colors(f,:), 'FaceAlpha', 0.7, 'EdgeColor', 'w');
        xline(median(vals), 'r--', 'LineWidth', 1.5);
        xlabel(param_names{p});
        ylabel('Count');
        title(sprintf('%s\nN=%d, med=%.3f', files{f,3}, length(vals), median(vals)));
        set(gca, 'FontSize', 10);
    end

    sgtitle(sprintf('Distribution of %s across subjects', param_names{p}), 'FontSize', 14);
end

%% Bonus: overlay all cohorts on one axis per parameter (softmax vs luce)
figure('Name', 'Overlay', 'Position', [100 100 1200 400]);

for p = 1:n_params
    subplot(1, n_params, p); hold on;

    for f = 1:n_files
        if isempty(all_params{f}), continue; end
        vals = all_params{f}(:, p);
        [counts, edges] = histcounts(vals, 15);
        centers = (edges(1:end-1) + edges(2:end)) / 2;
        plot(centers, counts, '-o', 'Color', colors(f,:), 'LineWidth', 1.5, ...
            'MarkerSize', 4, 'DisplayName', files{f,3});
    end

    xlabel(param_names{p});
    ylabel('Count');
    title(param_names{p});
    legend('Location', 'best', 'FontSize', 7);
    set(gca, 'FontSize', 11);
    hold off;
end

sgtitle('Parameter distributions: all models overlaid', 'FontSize', 14);