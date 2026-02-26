% Fit utility model to risk data
clear;clc

%% Cohort 1 Softmax
rdm_1 = load('rdm_1.mat');
rdm_1 = rdm_1.alldata;

% Fit RDM model
for s = 1:length(rdm_1)
    t = rdm_1(s).data;
    result_rdm = fitmodel_rdm_softmax(t);
    rdm_1(s).result_rdm = result_rdm;
    rdm_1(s).b_rdm      = result_rdm.b;
    rdm_1(s).pr2_rdm    = result_rdm.pseudoR2;
end

% Add model-free data
for s = 1:length(rdm_1)
    rdm_1(s).pGamble = mean(rdm_1(s).data(:,6));
end

% Save
save('fitrdmdata_softmax_1.mat', 'rdm_1'); % cohort 1

% Sanity check: alpha v pGam
B = vertcat(rdm_1.b_rdm);
pGamble = [rdm_1.pGamble]';

figure;
scatter(B(:,2), pGamble, 20, 'filled');
xlabel('alpha'); ylabel('pGamble');
[r,p] = corr(B(:,2), pGamble);
title(sprintf('Cohort 1 Softmax r=%.2f, p=%.3f', r, p));
box off; axis square;

% Sanity check: Parameter distributions
B = vertcat(rdm_1.b_rdm);
params = {'mu','alpha'};
figure;
for p = 1:2
    subplot(1,2,p);
    histogram(B(:,p), 20);
    xline(mean(B(:,p),   'omitnan'), 'r--', 'LineWidth', 1.5, 'Label', 'Mean');
    xline(median(B(:,p), 'omitnan'), 'b--', 'LineWidth', 1.5, 'Label', 'Median');
    title(params{p}); box off; axis square;
end
sgtitle('Cohort 1 Softmax RDM Model: Parameter distributions');

% Sanity check: pR2 distribution
figure('Color', 'w');
pr2_numeric = [rdm_1.pr2_rdm];
histogram(pr2_numeric, 80, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'w');
xlim([0,1])
xline(0, 'black', 'Chance', 'LineWidth', 2, 'LabelHorizontalAlignment', 'left');
xline(mean(pr2_numeric, 'omitnan'), '-b', sprintf('Mean (%.2f)', mean(pr2_numeric, 'omitnan')), 'LineWidth', 2);
xline(median(pr2_numeric, 'omitnan'), '-r', sprintf('Median (%.2f)', median(pr2_numeric, 'omitnan')), 'LineWidth', 2);
title('Cohort 1 Softmax Model Fit (pR^2)'); xlabel('Predictive Pseudo-R^2'); ylabel('Count'); axis square;
n_below_0  = sum(pr2_numeric < 0);
n_below_01 = sum(pr2_numeric < 0.1);

% Sanity check: Percent choice by SV difference bins
figure;
for s = 1:length(rdm_1)
    t = rdm_1(s).data;
    % SV difference: cert vs gamble (col3=cert, col4=gain, col5=pgain, col6=choice)
    sv_safe = t(:,3);
    sv_lott = t(:,4) .* t(:,5);   % gain * pgain
    sv_diff = sv_lott - sv_safe;
    choice  = t(:,6);
    rdm_1(s).sv_diff = sv_diff;
    rdm_1(s).choice  = choice;
end

all_sv  = vertcat(rdm_1.sv_diff);
all_ch  = vertcat(rdm_1.choice);

% Bin by SV difference
edges   = quantile(all_sv, linspace(0, 1, 11));
binIdx  = discretize(all_sv, edges);
pChoice = arrayfun(@(b) mean(all_ch(binIdx == b)), 1:10);
binCents = arrayfun(@(b) mean(all_sv(binIdx == b)), 1:10);
plot(binCents, pChoice, 'ko-', 'LineWidth', 2, 'MarkerFaceColor', 'k');
yline(0.5, 'k--'); ylim([0,1])
xlabel('SV difference (gamble - cert)'); ylabel('P(gamble)'); title('Cohort 1 Softmax Choice by SV difference (10 bins)');
axis square;

%% Cohort 1 Luce
rdm_1 = load('rdm_1.mat');
rdm_1 = rdm_1.alldata;

% Fit RDM model
for s = 1:length(rdm_1)
    t = rdm_1(s).data;
    result_rdm = fitmodel_rdm_luce(t);
    rdm_1(s).result_rdm = result_rdm;
    rdm_1(s).b_rdm      = result_rdm.b;
    rdm_1(s).pr2_rdm    = result_rdm.pseudoR2;
end

% Add model-free data
for s = 1:length(rdm_1)
    rdm_1(s).pGamble = mean(rdm_1(s).data(:,6));
end

% Save
save('fitrdmdata_luce_1.mat', 'rdm_1'); 

% Sanity check: alpha v pGam
B = vertcat(rdm_1.b_rdm);
pGamble = [rdm_1.pGamble]';

figure;
scatter(B(:,2), pGamble, 20, 'filled');
xlabel('alpha'); ylabel('pGamble');
[r,p] = corr(B(:,2), pGamble);
title(sprintf('Cohort 1 Luce r=%.2f, p=%.3f', r, p));
box off; axis square;

% Sanity check: Parameter distributions
B = vertcat(rdm_1.b_rdm);
params = {'mu','alpha'};
figure;
for p = 1:2
    subplot(1,2,p);
    histogram(B(:,p), 20);
    xline(mean(B(:,p),   'omitnan'), 'r--', 'LineWidth', 1.5, 'Label', 'Mean');
    xline(median(B(:,p), 'omitnan'), 'b--', 'LineWidth', 1.5, 'Label', 'Median');
    title(params{p}); box off; axis square;
end
sgtitle('Cohort 1 Luce RDM Model: Parameter distributions');

% Sanity check: pR2 distribution
figure('Color', 'w');
pr2_numeric = [rdm_1.pr2_rdm];
histogram(pr2_numeric, 80, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'w');
xlim([0,1])
xline(0, 'black', 'Chance', 'LineWidth', 2, 'LabelHorizontalAlignment', 'left');
xline(mean(pr2_numeric, 'omitnan'), '-b', sprintf('Mean (%.2f)', mean(pr2_numeric, 'omitnan')), 'LineWidth', 2);
xline(median(pr2_numeric, 'omitnan'), '-r', sprintf('Median (%.2f)', median(pr2_numeric, 'omitnan')), 'LineWidth', 2);
title('Cohort 1 Luce Model Fit (pR^2)'); xlabel('Predictive Pseudo-R^2'); ylabel('Count'); axis square;
n_below_0  = sum(pr2_numeric < 0);
n_below_01 = sum(pr2_numeric < 0.1);

% Sanity check: Percent choice by SV difference bins
figure;
for s = 1:length(rdm_1)
    t = rdm_1(s).data;
    % SV difference: cert vs gamble (col3=cert, col4=gain, col5=pgain, col6=choice)
    sv_safe = t(:,3);
    sv_lott = t(:,4) .* t(:,5);   % gain * pgain
    sv_diff = sv_lott - sv_safe;
    choice  = t(:,6);
    rdm_1(s).sv_diff = sv_diff;
    rdm_1(s).choice  = choice;
end

all_sv  = vertcat(rdm_1.sv_diff);
all_ch  = vertcat(rdm_1.choice);

% Bin by SV difference
edges   = quantile(all_sv, linspace(0, 1, 11));
binIdx  = discretize(all_sv, edges);
pChoice = arrayfun(@(b) mean(all_ch(binIdx == b)), 1:10);
binCents = arrayfun(@(b) mean(all_sv(binIdx == b)), 1:10);
plot(binCents, pChoice, 'ko-', 'LineWidth', 2, 'MarkerFaceColor', 'k');
yline(0.5, 'k--'); ylim([0,1])
xlabel('SV difference (gamble - cert)'); ylabel('P(gamble)'); title('Cohort 1 Luce Choice by SV difference (10 bins)');
axis square;

%% Cohort 2 Softmax
rdm_2 = load('rdm_2.mat');
rdm_2 = rdm_2.alldata;


% Fit RDM model
for s = 1:length(rdm_2)
    t = rdm_2(s).data;
    result_rdm = fitmodel_rdm_softmax(t);
    rdm_2(s).result_rdm = result_rdm;
    rdm_2(s).b_rdm      = result_rdm.b;
    rdm_2(s).pr2_rdm    = result_rdm.pseudoR2;
end

% Add model-free data
for s = 1:length(rdm_2)
    rdm_2(s).pGamble = mean(rdm_2(s).data(:,6));
end

% Save
save('fitrdmdata_softmax_2.mat', 'rdm_2');

% Sanity check: alpha v pGam
B = vertcat(rdm_2.b_rdm);
pGamble = [rdm_2.pGamble]';

figure;
scatter(B(:,2), pGamble, 20, 'filled');
xlabel('alpha'); ylabel('pGamble');
[r,p] = corr(B(:,2), pGamble);
title(sprintf('Cohort 2 Softmax r=%.2f, p=%.3f', r, p));
box off; axis square;

% Sanity check: Parameter distributions
B = vertcat(rdm_2.b_rdm);
params = {'mu','alpha'};
figure;
for p = 1:2
    subplot(1,2,p);
    histogram(B(:,p), 20);
    xline(mean(B(:,p),   'omitnan'), 'r--', 'LineWidth', 1.5, 'Label', 'Mean');
    xline(median(B(:,p), 'omitnan'), 'b--', 'LineWidth', 1.5, 'Label', 'Median');
    title(params{p}); box off; axis square;
end
sgtitle('Cohort 2 Softmax RDM Model: Parameter distributions');

% Sanity check: pR2 distribution
figure('Color', 'w');
pr2_numeric = [rdm_2.pr2_rdm];
histogram(pr2_numeric, 80, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'w');
xlim([0,1])
xline(0, 'black', 'Chance', 'LineWidth', 2, 'LabelHorizontalAlignment', 'left');
xline(mean(pr2_numeric, 'omitnan'), '-b', sprintf('Mean (%.2f)', mean(pr2_numeric, 'omitnan')), 'LineWidth', 2);
xline(median(pr2_numeric, 'omitnan'), '-r', sprintf('Median (%.2f)', median(pr2_numeric, 'omitnan')), 'LineWidth', 2);
title('Cohort 2 Softmax Model Fit (pR^2)'); xlabel('Predictive Pseudo-R^2'); ylabel('Count'); axis square;
n_below_0  = sum(pr2_numeric < 0);
n_below_01 = sum(pr2_numeric < 0.1);

% Sanity check: Percent choice by SV difference bins
figure;
for s = 1:length(rdm_2)
    t = rdm_2(s).data;
    % SV difference: cert vs gamble (col3=cert, col4=gain, col5=pgain, col6=choice)
    sv_safe = t(:,3);
    sv_lott = t(:,4) .* t(:,5);   % gain * pgain
    sv_diff = sv_lott - sv_safe;
    choice  = t(:,6);
    rdm_2(s).sv_diff = sv_diff;
    rdm_2(s).choice  = choice;
end

all_sv  = vertcat(rdm_2.sv_diff);
all_ch  = vertcat(rdm_2.choice);

% Bin by SV difference
edges   = quantile(all_sv, linspace(0, 1, 11));
binIdx  = discretize(all_sv, edges);
pChoice = arrayfun(@(b) mean(all_ch(binIdx == b)), 1:10);
binCents = arrayfun(@(b) mean(all_sv(binIdx == b)), 1:10);
plot(binCents, pChoice, 'ko-', 'LineWidth', 2, 'MarkerFaceColor', 'k');
yline(0.5, 'k--'); ylim([0,1])
xlabel('SV difference (gamble - cert)'); ylabel('P(gamble)'); title('Cohort 2 Softmax Choice by SV difference (10 bins)');
axis square;

%% Cohort 2 Luce
rdm_2 = load('rdm_2.mat');
rdm_2 = rdm_2.alldata;


% Fit RDM model
for s = 1:length(rdm_2)
    t = rdm_2(s).data;
    result_rdm = fitmodel_rdm_luce(t);
    rdm_2(s).result_rdm = result_rdm;
    rdm_2(s).b_rdm      = result_rdm.b;
    rdm_2(s).pr2_rdm    = result_rdm.pseudoR2;
end

% Add model-free data
for s = 1:length(rdm_2)
    rdm_2(s).pGamble = mean(rdm_2(s).data(:,6));
end

% Save
save('fitrdmdata_luce_2.mat', 'rdm_2');

% Sanity check: alpha v pGam
B = vertcat(rdm_2.b_rdm);
pGamble = [rdm_2.pGamble]';

figure;
scatter(B(:,2), pGamble, 20, 'filled');
xlabel('alpha'); ylabel('pGamble');
[r,p] = corr(B(:,2), pGamble);
title(sprintf('Cohort 2 Luce r=%.2f, p=%.3f', r, p));
box off; axis square;

% Sanity check: Parameter distributions
B = vertcat(rdm_2.b_rdm);
params = {'mu','alpha'};
figure;
for p = 1:2
    subplot(1,2,p);
    histogram(B(:,p), 20);
    xline(mean(B(:,p),   'omitnan'), 'r--', 'LineWidth', 1.5, 'Label', 'Mean');
    xline(median(B(:,p), 'omitnan'), 'b--', 'LineWidth', 1.5, 'Label', 'Median');
    title(params{p}); box off; axis square;
end
sgtitle('Cohort 2 Luce RDM Model: Parameter distributions');

% Sanity check: pR2 distribution
figure('Color', 'w');
pr2_numeric = [rdm_2.pr2_rdm];
histogram(pr2_numeric, 80, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'w');
xlim([0,1])
xline(0, 'black', 'Chance', 'LineWidth', 2, 'LabelHorizontalAlignment', 'left');
xline(mean(pr2_numeric, 'omitnan'), '-b', sprintf('Mean (%.2f)', mean(pr2_numeric, 'omitnan')), 'LineWidth', 2);
xline(median(pr2_numeric, 'omitnan'), '-r', sprintf('Median (%.2f)', median(pr2_numeric, 'omitnan')), 'LineWidth', 2);
title('Model Fit (pR^2)'); xlabel('Predictive Pseudo-R^2'); ylabel('Count'); axis square;
n_below_0  = sum(pr2_numeric < 0);
n_below_01 = sum(pr2_numeric < 0.1);

% Sanity check: Percent choice by SV difference bins
figure;
for s = 1:length(rdm_2)
    t = rdm_2(s).data;
    % SV difference: cert vs gamble (col3=cert, col4=gain, col5=pgain, col6=choice)
    sv_safe = t(:,3);
    sv_lott = t(:,4) .* t(:,5);   % gain * pgain
    sv_diff = sv_lott - sv_safe;
    choice  = t(:,6);
    rdm_2(s).sv_diff = sv_diff;
    rdm_2(s).choice  = choice;
end

all_sv  = vertcat(rdm_2.sv_diff);
all_ch  = vertcat(rdm_2.choice);

% Bin by SV difference
edges   = quantile(all_sv, linspace(0, 1, 11));
binIdx  = discretize(all_sv, edges);
pChoice = arrayfun(@(b) mean(all_ch(binIdx == b)), 1:10);
binCents = arrayfun(@(b) mean(all_sv(binIdx == b)), 1:10);
plot(binCents, pChoice, 'ko-', 'LineWidth', 2, 'MarkerFaceColor', 'k');
yline(0.5, 'k--'); ylim([0,1])
xlabel('SV difference (gamble - cert)'); ylabel('P(gamble)'); title('Cohort 2 Luce Choice by SV difference (10 bins)');
axis square;