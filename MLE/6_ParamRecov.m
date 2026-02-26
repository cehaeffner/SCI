% Run Parameter recovery
clear;clc;

%% Load fitted data
rdm_softmax_1 = load('fitrdmdata_softmax_1.mat'); rdm_softmax_1 = rdm_softmax_1.rdm_1;
rdm_softmax_2 = load('fitrdmdata_softmax_2.mat'); rdm_softmax_2 = rdm_softmax_2.rdm_2;
rdm_luce_1    = load('fitrdmdata_luce_1.mat');    rdm_luce_1    = rdm_luce_1.rdm_1;
rdm_luce_2    = load('fitrdmdata_luce_2.mat');    rdm_luce_2    = rdm_luce_2.rdm_2;
dd_softmax_1  = load('fitdddata_softmax_1.mat');  dd_softmax_1  = dd_softmax_1.dd_1;
dd_softmax_2  = load('fitdddata_softmax_2.mat');  dd_softmax_2  = dd_softmax_2.dd_2;
dd_luce_1     = load('fitdddata_luce_1.mat');     dd_luce_1     = dd_luce_1.dd_1;
dd_luce_2     = load('fitdddata_luce_2.mat');     dd_luce_2     = dd_luce_2.dd_2;

%% RDM Softmax Cohort 1
for s = 1:length(rdm_softmax_1)
    t          = rdm_softmax_1(s).data;
    mu_true    = rdm_softmax_1(s).b_rdm(1);
    alpha_true = rdm_softmax_1(s).b_rdm(2);
    sv_cert    = t(:,3).^alpha_true;
    sv_lott    = t(:,5) .* t(:,4).^alpha_true;
    utildiff   = sv_lott - sv_cert;
    probchoice = 1 ./ (1 + exp(-mu_true .* utildiff));
    t_sim      = t; t_sim(:,6) = double(rand(size(t,1),1) < probchoice);
    result_sim                    = fitmodel_rdm_softmax(t_sim);
    rdm_softmax_1(s).b_rdm_rec    = result_sim.b;
end
B_true = vertcat(rdm_softmax_1.b_rdm);
B_rec  = vertcat(rdm_softmax_1.b_rdm_rec);
params = {'mu','alpha'};
figure;
for p = 1:2
    subplot(1,2,p); scatter(B_true(:,p), B_rec(:,p), 40, 'filled'); hold on; refline(1,0);
    [r,pv] = corr(B_true(:,p), B_rec(:,p));
    xlabel(['true ' params{p}]); ylabel(['recovered ' params{p}]);
    title(sprintf('%s: r=%.2f, p=%.3f', params{p}, r, pv)); box off; axis square;
end
sgtitle('RDM Softmax - Cohort 1');

%% RDM Softmax Cohort 2
for s = 1:length(rdm_softmax_2)
    t          = rdm_softmax_2(s).data;
    mu_true    = rdm_softmax_2(s).b_rdm(1);
    alpha_true = rdm_softmax_2(s).b_rdm(2);
    sv_cert    = t(:,3).^alpha_true;
    sv_lott    = t(:,5) .* t(:,4).^alpha_true;
    utildiff   = sv_lott - sv_cert;
    probchoice = 1 ./ (1 + exp(-mu_true .* utildiff));
    t_sim      = t; t_sim(:,6) = double(rand(size(t,1),1) < probchoice);
    result_sim                    = fitmodel_rdm_softmax(t_sim);
    rdm_softmax_2(s).b_rdm_rec    = result_sim.b;
end
B_true = vertcat(rdm_softmax_2.b_rdm);
B_rec  = vertcat(rdm_softmax_2.b_rdm_rec);
figure;
for p = 1:2
    subplot(1,2,p); scatter(B_true(:,p), B_rec(:,p), 40, 'filled'); hold on; refline(1,0);
    [r,pv] = corr(B_true(:,p), B_rec(:,p));
    xlabel(['true ' params{p}]); ylabel(['recovered ' params{p}]);
    title(sprintf('%s: r=%.2f, p=%.3f', params{p}, r, pv)); box off; axis square;
end
sgtitle('RDM Softmax - Cohort 2');

%% RDM Luce Cohort 1
for s = 1:length(rdm_luce_1)
    t          = rdm_luce_1(s).data;
    mu_true    = rdm_luce_1(s).b_rdm(1);
    alpha_true = rdm_luce_1(s).b_rdm(2);
    sv_cert    = t(:,3).^alpha_true;
    sv_lott    = t(:,5) .* t(:,4).^alpha_true;
    probchoice = sv_lott.^mu_true ./ (sv_lott.^mu_true + sv_cert.^mu_true);
    t_sim      = t; t_sim(:,6) = double(rand(size(t,1),1) < probchoice);
    result_sim                 = fitmodel_rdm_luce(t_sim);
    rdm_luce_1(s).b_rdm_rec    = result_sim.b;
end
B_true = vertcat(rdm_luce_1.b_rdm);
B_rec  = vertcat(rdm_luce_1.b_rdm_rec);
figure;
for p = 1:2
    subplot(1,2,p); scatter(B_true(:,p), B_rec(:,p), 40, 'filled'); hold on; refline(1,0);
    [r,pv] = corr(B_true(:,p), B_rec(:,p));
    xlabel(['true ' params{p}]); ylabel(['recovered ' params{p}]);
    title(sprintf('%s: r=%.2f, p=%.3f', params{p}, r, pv)); box off; axis square;
end
sgtitle('RDM Luce - Cohort 1');

%% RDM Luce Cohort 2
for s = 1:length(rdm_luce_2)
    t          = rdm_luce_2(s).data;
    mu_true    = rdm_luce_2(s).b_rdm(1);
    alpha_true = rdm_luce_2(s).b_rdm(2);
    sv_cert    = t(:,3).^alpha_true;
    sv_lott    = t(:,5) .* t(:,4).^alpha_true;
    probchoice = sv_lott.^mu_true ./ (sv_lott.^mu_true + sv_cert.^mu_true);
    t_sim      = t; t_sim(:,6) = double(rand(size(t,1),1) < probchoice);
    result_sim                 = fitmodel_rdm_luce(t_sim);
    rdm_luce_2(s).b_rdm_rec    = result_sim.b;
end
B_true = vertcat(rdm_luce_2.b_rdm);
B_rec  = vertcat(rdm_luce_2.b_rdm_rec);
figure;
for p = 1:2
    subplot(1,2,p); scatter(B_true(:,p), B_rec(:,p), 40, 'filled'); hold on; refline(1,0);
    [r,pv] = corr(B_true(:,p), B_rec(:,p));
    xlabel(['true ' params{p}]); ylabel(['recovered ' params{p}]);
    title(sprintf('%s: r=%.2f, p=%.3f', params{p}, r, pv)); box off; axis square;
end
sgtitle('RDM Luce - Cohort 2');

%% DD Softmax Cohort 1
for s = 1:length(dd_softmax_1)
    t          = dd_softmax_1(s).data;
    mu_true    = dd_softmax_1(s).b_dd(1);
    kappa_true = dd_softmax_1(s).b_dd(2);
    alpha      = t(1,7);
    sv_sooner  = t(:,2).^alpha ./ (1 + kappa_true .* t(:,3));
    sv_later   = t(:,4).^alpha ./ (1 + kappa_true .* t(:,5));
    utildiff   = sv_later - sv_sooner;
    probchoice = 1 ./ (1 + exp(-mu_true .* utildiff));
    t_sim      = t; t_sim(:,6) = double(rand(size(t,1),1) < probchoice);
    result_sim                   = fitmodel_ddnlh_softmax(t_sim);
    dd_softmax_1(s).b_dd_rec     = result_sim.b;
end
B_true = vertcat(dd_softmax_1.b_dd);
B_rec  = vertcat(dd_softmax_1.b_dd_rec);
params = {'mu','kappa'};
figure;
for p = 1:2
    subplot(1,2,p); scatter(B_true(:,p), B_rec(:,p), 40, 'filled'); hold on; refline(1,0);
    [r,pv] = corr(B_true(:,p), B_rec(:,p));
    xlabel(['true ' params{p}]); ylabel(['recovered ' params{p}]);
    title(sprintf('%s: r=%.2f, p=%.3f', params{p}, r, pv)); box off; axis square;
end
sgtitle('DD Softmax - Cohort 1');

%% DD Softmax Cohort 2
for s = 1:length(dd_softmax_2)
    t          = dd_softmax_2(s).data;
    mu_true    = dd_softmax_2(s).b_dd(1);
    kappa_true = dd_softmax_2(s).b_dd(2);
    alpha      = t(1,7);
    sv_sooner  = t(:,2).^alpha ./ (1 + kappa_true .* t(:,3));
    sv_later   = t(:,4).^alpha ./ (1 + kappa_true .* t(:,5));
    utildiff   = sv_later - sv_sooner;
    probchoice = 1 ./ (1 + exp(-mu_true .* utildiff));
    t_sim      = t; t_sim(:,6) = double(rand(size(t,1),1) < probchoice);
    result_sim                   = fitmodel_ddnlh_softmax(t_sim);
    dd_softmax_2(s).b_dd_rec     = result_sim.b;
end
B_true = vertcat(dd_softmax_2.b_dd);
B_rec  = vertcat(dd_softmax_2.b_dd_rec);
figure;
for p = 1:2
    subplot(1,2,p); scatter(B_true(:,p), B_rec(:,p), 40, 'filled'); hold on; refline(1,0);
    [r,pv] = corr(B_true(:,p), B_rec(:,p));
    xlabel(['true ' params{p}]); ylabel(['recovered ' params{p}]);
    title(sprintf('%s: r=%.2f, p=%.3f', params{p}, r, pv)); box off; axis square;
end
sgtitle('DD Softmax - Cohort 2');

%% DD Luce Cohort 1
for s = 1:length(dd_luce_1)
    t          = dd_luce_1(s).data;
    mu_true    = dd_luce_1(s).b_dd(1);
    kappa_true = dd_luce_1(s).b_dd(2);
    alpha      = t(1,7);
    sv_sooner  = t(:,2).^alpha ./ (1 + kappa_true .* t(:,3));
    sv_later   = t(:,4).^alpha ./ (1 + kappa_true .* t(:,5));
    probchoice = sv_later.^mu_true ./ (sv_later.^mu_true + sv_sooner.^mu_true);
    t_sim      = t; t_sim(:,6) = double(rand(size(t,1),1) < probchoice);
    result_sim               = fitmodel_ddnlh_luce(t_sim);
    dd_luce_1(s).b_dd_rec    = result_sim.b;
end
B_true = vertcat(dd_luce_1.b_dd);
B_rec  = vertcat(dd_luce_1.b_dd_rec);
figure;
for p = 1:2
    subplot(1,2,p); scatter(B_true(:,p), B_rec(:,p), 40, 'filled'); hold on; refline(1,0);
    [r,pv] = corr(B_true(:,p), B_rec(:,p));
    xlabel(['true ' params{p}]); ylabel(['recovered ' params{p}]);
    title(sprintf('%s: r=%.2f, p=%.3f', params{p}, r, pv)); box off; axis square;
end
sgtitle('DD Luce - Cohort 1');

%% DD Luce Cohort 2
for s = 1:length(dd_luce_2)
    t          = dd_luce_2(s).data;
    mu_true    = dd_luce_2(s).b_dd(1);
    kappa_true = dd_luce_2(s).b_dd(2);
    alpha      = t(1,7);
    sv_sooner  = t(:,2).^alpha ./ (1 + kappa_true .* t(:,3));
    sv_later   = t(:,4).^alpha ./ (1 + kappa_true .* t(:,5));
    probchoice = sv_later.^mu_true ./ (sv_later.^mu_true + sv_sooner.^mu_true);
    t_sim      = t; t_sim(:,6) = double(rand(size(t,1),1) < probchoice);
    result_sim               = fitmodel_ddnlh_luce(t_sim);
    dd_luce_2(s).b_dd_rec    = result_sim.b;
end
B_true = vertcat(dd_luce_2.b_dd);
B_rec  = vertcat(dd_luce_2.b_dd_rec);
figure;
for p = 1:2
    subplot(1,2,p); scatter(B_true(:,p), B_rec(:,p), 40, 'filled'); hold on; refline(1,0);
    [r,pv] = corr(B_true(:,p), B_rec(:,p));
    xlabel(['true ' params{p}]); ylabel(['recovered ' params{p}]);
    title(sprintf('%s: r=%.2f, p=%.3f', params{p}, r, pv)); box off; axis square;
end
sgtitle('DD Luce - Cohort 2');