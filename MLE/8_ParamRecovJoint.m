% Parameter recovery on joint data
clear;clc;

%% Load fitted data
joint_softmax_1 = load('fitjointdata_softmax_1.mat'); joint_softmax_1 = joint_softmax_1.rdm_1;
joint_luce_1    = load('fitjointdata_luce_1.mat');    joint_luce_1    = joint_luce_1.rdm_1;
joint_softmax_2 = load('fitjointdata_softmax_2.mat'); joint_softmax_2 = joint_softmax_2.rdm_2;
joint_luce_2    = load('fitjointdata_luce_2.mat');    joint_luce_2    = joint_luce_2.rdm_2;

% Load DD data (to get t_dd for simulation)
dd_softmax_1 = load('dd_softmax_1.mat'); dd_softmax_1 = dd_softmax_1.alldata;
dd_luce_1    = load('dd_luce_1.mat');    dd_luce_1    = dd_luce_1.alldata;
dd_softmax_2 = load('dd_softmax_2.mat'); dd_softmax_2 = dd_softmax_2.alldata;
dd_luce_2    = load('dd_luce_2.mat');    dd_luce_2    = dd_luce_2.alldata;

%% Joint Softmax Cohort 1
dd_subids = [dd_softmax_1.subid]';
for s = 1:length(joint_softmax_1)
    dd_idx = find(dd_subids == joint_softmax_1(s).subid);
    if isempty(dd_idx); continue; end
    t_rdm      = joint_softmax_1(s).data;
    t_dd       = dd_softmax_1(dd_idx).data;
    mu_true    = joint_softmax_1(s).b_joint(1);
    alpha_true = joint_softmax_1(s).b_joint(2);
    kappa_true = joint_softmax_1(s).b_joint(3);
    % Simulate RDM
    sv_cert        = t_rdm(:,3).^alpha_true;
    sv_lott        = t_rdm(:,5) .* t_rdm(:,4).^alpha_true;
    prob_rdm       = 1 ./ (1 + exp(-mu_true .* (sv_lott - sv_cert)));
    t_rdm_sim      = t_rdm; t_rdm_sim(:,6) = double(rand(size(t_rdm,1),1) < prob_rdm);
    % Simulate DD
    alpha_dd       = t_dd(:,7);
    sv_sooner      = t_dd(:,2).^alpha_dd ./ (1 + kappa_true .* t_dd(:,3));
    sv_later       = t_dd(:,4).^alpha_dd ./ (1 + kappa_true .* t_dd(:,5));
    prob_dd        = 1 ./ (1 + exp(-mu_true .* (sv_later - sv_sooner)));
    t_dd_sim       = t_dd; t_dd_sim(:,6) = double(rand(size(t_dd,1),1) < prob_dd);
    result_sim                        = fitmodel_joint_softmax(t_rdm_sim, t_dd_sim);
    joint_softmax_1(s).b_joint_rec    = result_sim.b;
end
B_true = vertcat(joint_softmax_1.b_joint);
B_rec  = vertcat(joint_softmax_1.b_joint_rec);
params = {'mu','alpha','kappa'};
figure;
for p = 1:3
    subplot(1,3,p); scatter(B_true(:,p), B_rec(:,p), 40, 'filled'); hold on; refline(1,0);
    [r,pv] = corr(B_true(:,p), B_rec(:,p));
    xlabel(['true ' params{p}]); ylabel(['recovered ' params{p}]);
    title(sprintf('%s: r=%.2f, p=%.3f', params{p}, r, pv)); box off; axis square;
end
sgtitle('Joint Softmax - Cohort 1');

%% Joint Softmax Cohort 2
dd_subids = [dd_softmax_2.subid]';
for s = 1:length(joint_softmax_2)
    dd_idx = find(dd_subids == joint_softmax_2(s).subid);
    if isempty(dd_idx); continue; end
    t_rdm      = joint_softmax_2(s).data;
    t_dd       = dd_softmax_2(dd_idx).data;
    mu_true    = joint_softmax_2(s).b_joint(1);
    alpha_true = joint_softmax_2(s).b_joint(2);
    kappa_true = joint_softmax_2(s).b_joint(3);
    sv_cert        = t_rdm(:,3).^alpha_true;
    sv_lott        = t_rdm(:,5) .* t_rdm(:,4).^alpha_true;
    prob_rdm       = 1 ./ (1 + exp(-mu_true .* (sv_lott - sv_cert)));
    t_rdm_sim      = t_rdm; t_rdm_sim(:,6) = double(rand(size(t_rdm,1),1) < prob_rdm);
    alpha_dd       = t_dd(:,7);
    sv_sooner      = t_dd(:,2).^alpha_dd ./ (1 + kappa_true .* t_dd(:,3));
    sv_later       = t_dd(:,4).^alpha_dd ./ (1 + kappa_true .* t_dd(:,5));
    prob_dd        = 1 ./ (1 + exp(-mu_true .* (sv_later - sv_sooner)));
    t_dd_sim       = t_dd; t_dd_sim(:,6) = double(rand(size(t_dd,1),1) < prob_dd);
    result_sim                        = fitmodel_joint_softmax(t_rdm_sim, t_dd_sim);
    joint_softmax_2(s).b_joint_rec    = result_sim.b;
end
B_true = vertcat(joint_softmax_2.b_joint);
B_rec  = vertcat(joint_softmax_2.b_joint_rec);
figure;
for p = 1:3
    subplot(1,3,p); scatter(B_true(:,p), B_rec(:,p), 40, 'filled'); hold on; refline(1,0);
    [r,pv] = corr(B_true(:,p), B_rec(:,p));
    xlabel(['true ' params{p}]); ylabel(['recovered ' params{p}]);
    title(sprintf('%s: r=%.2f, p=%.3f', params{p}, r, pv)); box off; axis square;
end
sgtitle('Joint Softmax - Cohort 2');

%% Joint Luce Cohort 1
dd_subids = [dd_luce_1.subid]';
for s = 1:length(joint_luce_1)
    dd_idx = find(dd_subids == joint_luce_1(s).subid);
    if isempty(dd_idx); continue; end
    t_rdm      = joint_luce_1(s).data;
    t_dd       = dd_luce_1(dd_idx).data;
    mu_true    = joint_luce_1(s).b_joint(1);
    alpha_true = joint_luce_1(s).b_joint(2);
    kappa_true = joint_luce_1(s).b_joint(3);
    % Simulate RDM (Luce)
    sv_cert        = t_rdm(:,3).^alpha_true;
    sv_lott        = t_rdm(:,5) .* t_rdm(:,4).^alpha_true;
    prob_rdm       = sv_lott.^mu_true ./ (sv_lott.^mu_true + sv_cert.^mu_true);
    t_rdm_sim      = t_rdm; t_rdm_sim(:,6) = double(rand(size(t_rdm,1),1) < prob_rdm);
    % Simulate DD (Luce)
    alpha_dd       = t_dd(:,7);
    sv_sooner      = t_dd(:,2).^alpha_dd ./ (1 + kappa_true .* t_dd(:,3));
    sv_later       = t_dd(:,4).^alpha_dd ./ (1 + kappa_true .* t_dd(:,5));
    prob_dd        = sv_later.^mu_true ./ (sv_later.^mu_true + sv_sooner.^mu_true);
    t_dd_sim       = t_dd; t_dd_sim(:,6) = double(rand(size(t_dd,1),1) < prob_dd);
    result_sim                     = fitmodel_joint_luce(t_rdm_sim, t_dd_sim);
    joint_luce_1(s).b_joint_rec    = result_sim.b;
end
B_true = vertcat(joint_luce_1.b_joint);
B_rec  = vertcat(joint_luce_1.b_joint_rec);
figure;
for p = 1:3
    subplot(1,3,p); scatter(B_true(:,p), B_rec(:,p), 40, 'filled'); hold on; refline(1,0);
    [r,pv] = corr(B_true(:,p), B_rec(:,p));
    xlabel(['true ' params{p}]); ylabel(['recovered ' params{p}]);
    title(sprintf('%s: r=%.2f, p=%.3f', params{p}, r, pv)); box off; axis square;
end
sgtitle('Joint Luce - Cohort 1');

%% Joint Luce Cohort 2
dd_subids = [dd_luce_2.subid]';
for s = 1:length(joint_luce_2)
    dd_idx = find(dd_subids == joint_luce_2(s).subid);
    if isempty(dd_idx); continue; end
    t_rdm      = joint_luce_2(s).data;
    t_dd       = dd_luce_2(dd_idx).data;
    mu_true    = joint_luce_2(s).b_joint(1);
    alpha_true = joint_luce_2(s).b_joint(2);
    kappa_true = joint_luce_2(s).b_joint(3);
    sv_cert        = t_rdm(:,3).^alpha_true;
    sv_lott        = t_rdm(:,5) .* t_rdm(:,4).^alpha_true;
    prob_rdm       = sv_lott.^mu_true ./ (sv_lott.^mu_true + sv_cert.^mu_true);
    t_rdm_sim      = t_rdm; t_rdm_sim(:,6) = double(rand(size(t_rdm,1),1) < prob_rdm);
    alpha_dd       = t_dd(:,7);
    sv_sooner      = t_dd(:,2).^alpha_dd ./ (1 + kappa_true .* t_dd(:,3));
    sv_later       = t_dd(:,4).^alpha_dd ./ (1 + kappa_true .* t_dd(:,5));
    prob_dd        = sv_later.^mu_true ./ (sv_later.^mu_true + sv_sooner.^mu_true);
    t_dd_sim       = t_dd; t_dd_sim(:,6) = double(rand(size(t_dd,1),1) < prob_dd);
    result_sim                     = fitmodel_joint_luce(t_rdm_sim, t_dd_sim);
    joint_luce_2(s).b_joint_rec    = result_sim.b;
end
B_true = vertcat(joint_luce_2.b_joint);
B_rec  = vertcat(joint_luce_2.b_joint_rec);
figure;
for p = 1:3
    subplot(1,3,p); scatter(B_true(:,p), B_rec(:,p), 40, 'filled'); hold on; refline(1,0);
    [r,pv] = corr(B_true(:,p), B_rec(:,p));
    xlabel(['true ' params{p}]); ylabel(['recovered ' params{p}]);
    title(sprintf('%s: r=%.2f, p=%.3f', params{p}, r, pv)); box off; axis square;
end
sgtitle('Joint Luce - Cohort 2');