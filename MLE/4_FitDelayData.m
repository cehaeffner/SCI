% Fit nonlinear hyperbolic discounting model to delay data
clear;clc

%% Cohort 1 Softmax
dd_1 = load('dd_softmax_1.mat');
dd_1 = dd_1.alldata;

% Fit DD model
for s = 1:length(dd_1)
    t = dd_1(s).data;
    result_dd = fitmodel_ddnlh_softmax(t);
    dd_1(s).result_dd = result_dd;
    dd_1(s).b_dd      = result_dd.b;
    dd_1(s).pr2_dd    = result_dd.pseudoR2;
end

% Add model-free data
for s = 1:length(dd_1)
    dd_1(s).pLater = mean(dd_1(s).data(:,6));  % col 6: 1=later, 0=sooner
end

% Save
save('fitdddata_softmax_1.mat', 'dd_1');

% Sanity check: kappa v pLater
B = vertcat(dd_1.b_dd);
pLater = [dd_1.pLater]';
figure;
scatter(B(:,2), pLater, 20, 'filled');
xlabel('kappa'); ylabel('pLater');
[r,p] = corr(B(:,2), pLater);
title(sprintf('Cohort 1 Softmax r=%.2f, p=%.3f', r, p));
box off; axis square;

% Sanity check: Parameter distributions
B = vertcat(dd_1.b_dd);
params = {'mu','kappa'};
figure;
for p = 1:2
    subplot(1,2,p);
    histogram(B(:,p), 20);
    xline(mean(B(:,p),   'omitnan'), 'r--', 'LineWidth', 1.5, 'Label', 'Mean');
    xline(median(B(:,p), 'omitnan'), 'b--', 'LineWidth', 1.5, 'Label', 'Median');
    title(params{p}); box off; axis square;
end
sgtitle('Cohort 1 Softmax DD Model: Parameter distributions');

% Sanity check: pR2 distribution
figure('Color', 'w');
pr2_numeric = [dd_1.pr2_dd];
histogram(pr2_numeric, 80, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'w');
xlim([0,1])
xline(0, 'black', 'Chance', 'LineWidth', 2, 'LabelHorizontalAlignment', 'left');
xline(mean(pr2_numeric, 'omitnan'), '-b', sprintf('Mean (%.2f)', mean(pr2_numeric, 'omitnan')), 'LineWidth', 2);
xline(median(pr2_numeric, 'omitnan'), '-r', sprintf('Median (%.2f)', median(pr2_numeric, 'omitnan')), 'LineWidth', 2);
title('Cohort 1 Softmax Model Fit (pR^2)'); xlabel('Predictive Pseudo-R^2'); ylabel('Count'); axis square;
n_below_0  = sum(pr2_numeric < 0);
n_below_01 = sum(pr2_numeric < 0.1);

% Sanity check: Percent choice by SV difference bins
% SV: amount/(1+kappa*delay), col2=sooner amt, col3=sooner delay,
%     col4=later amt, col5=later delay, col6=choice, col7=alpha
figure;
for s = 1:length(dd_1)
    t     = dd_1(s).data;
    kappa = dd_1(s).b_dd(2);
    alpha = t(1,7);  % same for all trials
    sv_sooner = t(:,2).^alpha ./ (1 + kappa .* t(:,3));
    sv_later  = t(:,4).^alpha ./ (1 + kappa .* t(:,5));
    sv_diff   = sv_later - sv_sooner;
    choice    = t(:,6);
    dd_1(s).sv_diff = sv_diff;
    dd_1(s).choice  = choice;
end

all_sv  = vertcat(dd_1.sv_diff);
all_ch  = vertcat(dd_1.choice);

edges    = quantile(all_sv, linspace(0, 1, 11));
binIdx   = discretize(all_sv, edges);
pChoice  = arrayfun(@(b) mean(all_ch(binIdx == b)), 1:10);
binCents = arrayfun(@(b) mean(all_sv(binIdx == b)), 1:10);

plot(binCents, pChoice, 'ko-', 'LineWidth', 2, 'MarkerFaceColor', 'k');
yline(0.5, 'k--'); ylim([0,1])
xlabel('SV difference (later - sooner)'); ylabel('P(later)'); title('Cohort 1 Softmax Choice by SV difference (10 bins)');
axis square;

%% Cohort 1 Luce
dd_1 = load('dd_luce_1.mat');
dd_1 = dd_1.alldata;

% Fit DD model
for s = 1:length(dd_1)
    t = dd_1(s).data;
    result_dd = fitmodel_ddnlh_luce(t);
    dd_1(s).result_dd = result_dd;
    dd_1(s).b_dd      = result_dd.b;
    dd_1(s).pr2_dd    = result_dd.pseudoR2;
end

% Add model-free data
for s = 1:length(dd_1)
    dd_1(s).pLater = mean(dd_1(s).data(:,6));  % col 6: 1=later, 0=sooner
end

% Save
save('fitdddata_luce_1.mat', 'dd_1');

% Sanity check: kappa v pLater
B = vertcat(dd_1.b_dd);
pLater = [dd_1.pLater]';
figure;
scatter(B(:,2), pLater, 20, 'filled');
xlabel('kappa'); ylabel('pLater');
[r,p] = corr(B(:,2), pLater);
title(sprintf('Cohort 1 Luce r=%.2f, p=%.3f', r, p));
box off; axis square;

% Sanity check: Parameter distributions
B = vertcat(dd_1.b_dd);
params = {'mu','kappa'};
figure;
for p = 1:2
    subplot(1,2,p);
    histogram(B(:,p), 20);
    xline(mean(B(:,p),   'omitnan'), 'r--', 'LineWidth', 1.5, 'Label', 'Mean');
    xline(median(B(:,p), 'omitnan'), 'b--', 'LineWidth', 1.5, 'Label', 'Median');
    title(params{p}); box off; axis square;
end
sgtitle('Cohort 1 Luce DD Model: Parameter distributions');

% Sanity check: pR2 distribution
figure('Color', 'w');
pr2_numeric = [dd_1.pr2_dd];
histogram(pr2_numeric, 80, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'w');
xlim([0,1])
xline(0, 'black', 'Chance', 'LineWidth', 2, 'LabelHorizontalAlignment', 'left');
xline(mean(pr2_numeric, 'omitnan'), '-b', sprintf('Mean (%.2f)', mean(pr2_numeric, 'omitnan')), 'LineWidth', 2);
xline(median(pr2_numeric, 'omitnan'), '-r', sprintf('Median (%.2f)', median(pr2_numeric, 'omitnan')), 'LineWidth', 2);
title('Cohort 1 Luce Model Fit (pR^2)'); xlabel('Predictive Pseudo-R^2'); ylabel('Count'); axis square;
n_below_0  = sum(pr2_numeric < 0);
n_below_01 = sum(pr2_numeric < 0.1);

% Sanity check: Percent choice by SV difference bins
% SV: amount/(1+kappa*delay), col2=sooner amt, col3=sooner delay,
%     col4=later amt, col5=later delay, col6=choice, col7=alpha
figure;
for s = 1:length(dd_1)
    t     = dd_1(s).data;
    kappa = dd_1(s).b_dd(2);
    alpha = t(1,7);  % same for all trials
    sv_sooner = t(:,2).^alpha ./ (1 + kappa .* t(:,3));
    sv_later  = t(:,4).^alpha ./ (1 + kappa .* t(:,5));
    sv_diff   = sv_later - sv_sooner;
    choice    = t(:,6);
    dd_1(s).sv_diff = sv_diff;
    dd_1(s).choice  = choice;
end

all_sv  = vertcat(dd_1.sv_diff);
all_ch  = vertcat(dd_1.choice);

edges    = quantile(all_sv, linspace(0, 1, 11));
binIdx   = discretize(all_sv, edges);
pChoice  = arrayfun(@(b) mean(all_ch(binIdx == b)), 1:10);
binCents = arrayfun(@(b) mean(all_sv(binIdx == b)), 1:10);

plot(binCents, pChoice, 'ko-', 'LineWidth', 2, 'MarkerFaceColor', 'k');
yline(0.5, 'k--'); ylim([0,1])
xlabel('SV difference (later - sooner)'); ylabel('P(later)'); title('Cohort 1 Luce Choice by SV difference (10 bins)');
axis square;

%% Cohort 2 Softmax
dd_2 = load('dd_softmax_2.mat');
dd_2 = dd_2.alldata;

% Fit DD model
for s = 1:length(dd_2)
    t = dd_2(s).data;
    result_dd = fitmodel_ddnlh_softmax(t);
    dd_2(s).result_dd = result_dd;
    dd_2(s).b_dd      = result_dd.b;
    dd_2(s).pr2_dd    = result_dd.pseudoR2;
end

% Add model-free data
for s = 1:length(dd_2)
    dd_2(s).pLater = mean(dd_2(s).data(:,6));  % col 6: 1=later, 0=sooner
end

% Save
save('fitdddata_softmax_2.mat', 'dd_2');

% Sanity check: kappa v pLater
B = vertcat(dd_2.b_dd);
pLater = [dd_2.pLater]';
figure;
scatter(B(:,2), pLater, 20, 'filled');
xlabel('kappa'); ylabel('pLater');
[r,p] = corr(B(:,2), pLater);
title(sprintf('Cohort 2 Softmax r=%.2f, p=%.3f', r, p));
box off; axis square;

% Sanity check: Parameter distributions
B = vertcat(dd_2.b_dd);
params = {'mu','kappa'};
figure;
for p = 1:2
    subplot(1,2,p);
    histogram(B(:,p), 20);
    xline(mean(B(:,p),   'omitnan'), 'r--', 'LineWidth', 1.5, 'Label', 'Mean');
    xline(median(B(:,p), 'omitnan'), 'b--', 'LineWidth', 1.5, 'Label', 'Median');
    title(params{p}); box off; axis square;
end
sgtitle('Cohort 2 Softmax DD Model: Parameter distributions');

% Sanity check: pR2 distribution
figure('Color', 'w');
pr2_numeric = [dd_2.pr2_dd];
histogram(pr2_numeric, 80, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'w');
xlim([0,1])
xline(0, 'black', 'Chance', 'LineWidth', 2, 'LabelHorizontalAlignment', 'left');
xline(mean(pr2_numeric, 'omitnan'), '-b', sprintf('Mean (%.2f)', mean(pr2_numeric, 'omitnan')), 'LineWidth', 2);
xline(median(pr2_numeric, 'omitnan'), '-r', sprintf('Median (%.2f)', median(pr2_numeric, 'omitnan')), 'LineWidth', 2);
title('Cohort 2 Softmax Model Fit (pR^2)'); xlabel('Predictive Pseudo-R^2'); ylabel('Count'); axis square;
n_below_0  = sum(pr2_numeric < 0);
n_below_01 = sum(pr2_numeric < 0.1);

% Sanity check: Percent choice by SV difference bins
% SV: amount/(1+kappa*delay), col2=sooner amt, col3=sooner delay,
%     col4=later amt, col5=later delay, col6=choice, col7=alpha
figure;
for s = 1:length(dd_2)
    t     = dd_2(s).data;
    kappa = dd_2(s).b_dd(2);
    alpha = t(1,7);  % same for all trials
    sv_sooner = t(:,2).^alpha ./ (1 + kappa .* t(:,3));
    sv_later  = t(:,4).^alpha ./ (1 + kappa .* t(:,5));
    sv_diff   = sv_later - sv_sooner;
    choice    = t(:,6);
    dd_2(s).sv_diff = sv_diff;
    dd_2(s).choice  = choice;
end

all_sv  = vertcat(dd_2.sv_diff);
all_ch  = vertcat(dd_2.choice);

edges    = quantile(all_sv, linspace(0, 1, 11));
binIdx   = discretize(all_sv, edges);
pChoice  = arrayfun(@(b) mean(all_ch(binIdx == b)), 1:10);
binCents = arrayfun(@(b) mean(all_sv(binIdx == b)), 1:10);

plot(binCents, pChoice, 'ko-', 'LineWidth', 2, 'MarkerFaceColor', 'k');
yline(0.5, 'k--'); ylim([0,1])
xlabel('SV difference (later - sooner)'); ylabel('P(later)'); title('Cohort 2 Softmax Choice by SV difference (10 bins)');
axis square;

%% Cohort 2 Luce
dd_2 = load('dd_luce_2.mat');
dd_2 = dd_2.alldata;

% Fit DD model
for s = 1:length(dd_2)
    t = dd_2(s).data;
    result_dd = fitmodel_ddnlh_luce(t);
    dd_2(s).result_dd = result_dd;
    dd_2(s).b_dd      = result_dd.b;
    dd_2(s).pr2_dd    = result_dd.pseudoR2;
end

% Add model-free data
for s = 1:length(dd_2)
    dd_2(s).pLater = mean(dd_2(s).data(:,6));  % col 6: 1=later, 0=sooner
end

% Save
save('fitdddata_luce_2.mat', 'dd_2');

% Sanity check: kappa v pLater
B = vertcat(dd_2.b_dd);
pLater = [dd_2.pLater]';
figure;
scatter(B(:,2), pLater, 20, 'filled');
xlabel('kappa'); ylabel('pLater');
[r,p] = corr(B(:,2), pLater);
title(sprintf('Cohort 2 Luce r=%.2f, p=%.3f', r, p));
box off; axis square;

% Sanity check: Parameter distributions
B = vertcat(dd_2.b_dd);
params = {'mu','kappa'};
figure;
for p = 1:2
    subplot(1,2,p);
    histogram(B(:,p), 20);
    xline(mean(B(:,p),   'omitnan'), 'r--', 'LineWidth', 1.5, 'Label', 'Mean');
    xline(median(B(:,p), 'omitnan'), 'b--', 'LineWidth', 1.5, 'Label', 'Median');
    title(params{p}); box off; axis square;
end
sgtitle('Cohort 2 Luce DD Model: Parameter distributions');

% Sanity check: pR2 distribution
figure('Color', 'w');
pr2_numeric = [dd_2.pr2_dd];
histogram(pr2_numeric, 80, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'w');
xlim([0,1])
xline(0, 'black', 'Chance', 'LineWidth', 2, 'LabelHorizontalAlignment', 'left');
xline(mean(pr2_numeric, 'omitnan'), '-b', sprintf('Mean (%.2f)', mean(pr2_numeric, 'omitnan')), 'LineWidth', 2);
xline(median(pr2_numeric, 'omitnan'), '-r', sprintf('Median (%.2f)', median(pr2_numeric, 'omitnan')), 'LineWidth', 2);
title('Cohort 2 Luce Model Fit (pR^2)'); xlabel('Predictive Pseudo-R^2'); ylabel('Count'); axis square;
n_below_0  = sum(pr2_numeric < 0);
n_below_01 = sum(pr2_numeric < 0.1);

% Sanity check: Percent choice by SV difference bins
% SV: amount/(1+kappa*delay), col2=sooner amt, col3=sooner delay,
%     col4=later amt, col5=later delay, col6=choice, col7=alpha
figure;
for s = 1:length(dd_2)
    t     = dd_2(s).data;
    kappa = dd_2(s).b_dd(2);
    alpha = t(1,7);  % same for all trials
    sv_sooner = t(:,2).^alpha ./ (1 + kappa .* t(:,3));
    sv_later  = t(:,4).^alpha ./ (1 + kappa .* t(:,5));
    sv_diff   = sv_later - sv_sooner;
    choice    = t(:,6);
    dd_2(s).sv_diff = sv_diff;
    dd_2(s).choice  = choice;
end

all_sv  = vertcat(dd_2.sv_diff);
all_ch  = vertcat(dd_2.choice);

edges    = quantile(all_sv, linspace(0, 1, 11));
binIdx   = discretize(all_sv, edges);
pChoice  = arrayfun(@(b) mean(all_ch(binIdx == b)), 1:10);
binCents = arrayfun(@(b) mean(all_sv(binIdx == b)), 1:10);

plot(binCents, pChoice, 'ko-', 'LineWidth', 2, 'MarkerFaceColor', 'k');
yline(0.5, 'k--'); ylim([0,1])
xlabel('SV difference (later - sooner)'); ylabel('P(later)'); title('Cohort 2 Luce Choice by SV difference (10 bins)');
axis square;