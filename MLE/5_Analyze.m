% Analyze cross-task mu fits
clear;clc;

% Load
rdm_softmax_1 = load("fitrdmdata_softmax_1.mat"); rdm_softmax_1 = rdm_softmax_1.rdm_1;
rdm_luce_1    = load("fitrdmdata_luce_1.mat");    rdm_luce_1    = rdm_luce_1.rdm_1;
rdm_softmax_2 = load("fitrdmdata_softmax_2.mat"); rdm_softmax_2 = rdm_softmax_2.rdm_2;
rdm_luce_2    = load("fitrdmdata_luce_2.mat");    rdm_luce_2    = rdm_luce_2.rdm_2;
dd_softmax_1  = load("fitdddata_softmax_1.mat");  dd_softmax_1  = dd_softmax_1.dd_1;
dd_luce_1     = load("fitdddata_luce_1.mat");     dd_luce_1     = dd_luce_1.dd_1;
dd_softmax_2  = load("fitdddata_softmax_2.mat");  dd_softmax_2  = dd_softmax_2.dd_2;
dd_luce_2     = load("fitdddata_luce_2.mat");     dd_luce_2     = dd_luce_2.dd_2;

% Softmax Cohort 1
rdm_t = struct2table(rdm_softmax_1); rdm_t = renamevars(rdm_t, 'data', 'data_rdm');
dd_t  = struct2table(dd_softmax_1);  dd_t  = renamevars(dd_t,  'data', 'data_dd');
data  = innerjoin(rdm_t, dd_t, 'Keys', 'subid');
B_rdm = vertcat(data.b_rdm); B_dd = vertcat(data.b_dd);
figure;
scatter(B_rdm(:,1), B_dd(:,1), 40, 'm', 'filled');
xlabel('mu (RDM)'); ylabel('mu (DD)');
[r,p] = corr(B_rdm(:,1), B_dd(:,1));
title(sprintf('Softmax Cohort 1 | r=%.2f, p=%.3f', r, p));
box off; axis square; refline(1,0);

% Softmax Cohort 2
rdm_t = struct2table(rdm_softmax_2); rdm_t = renamevars(rdm_t, 'data', 'data_rdm');
dd_t  = struct2table(dd_softmax_2);  dd_t  = renamevars(dd_t,  'data', 'data_dd');
data  = innerjoin(rdm_t, dd_t, 'Keys', 'subid');
B_rdm = vertcat(data.b_rdm); B_dd = vertcat(data.b_dd);
figure;
scatter(B_rdm(:,1), B_dd(:,1), 40, 'm', 'filled');
xlabel('mu (RDM)'); ylabel('mu (DD)');
[r,p] = corr(B_rdm(:,1), B_dd(:,1));
title(sprintf('Softmax Cohort 2 | r=%.2f, p=%.3f', r, p));
box off; axis square; refline(1,0);

% Luce Cohort 1
rdm_t = struct2table(rdm_luce_1); rdm_t = renamevars(rdm_t, 'data', 'data_rdm');
dd_t  = struct2table(dd_luce_1);  dd_t  = renamevars(dd_t,  'data', 'data_dd');
data  = innerjoin(rdm_t, dd_t, 'Keys', 'subid');
B_rdm = vertcat(data.b_rdm); B_dd = vertcat(data.b_dd);
figure;
scatter(B_rdm(:,1), B_dd(:,1), 40, 'b', 'filled');
xlabel('mu (RDM)'); ylabel('mu (DD)');
[r,p] = corr(B_rdm(:,1), B_dd(:,1));
title(sprintf('Luce Cohort 1 | r=%.2f, p=%.3f', r, p));
box off; axis square; refline(1,0);

% Luce Cohort 2
rdm_t = struct2table(rdm_luce_2); rdm_t = renamevars(rdm_t, 'data', 'data_rdm');
dd_t  = struct2table(dd_luce_2);  dd_t  = renamevars(dd_t,  'data', 'data_dd');
data  = innerjoin(rdm_t, dd_t, 'Keys', 'subid');
B_rdm = vertcat(data.b_rdm); B_dd = vertcat(data.b_dd);
figure;
scatter(B_rdm(:,1), B_dd(:,1), 40, 'b', 'filled');
xlabel('mu (RDM)'); ylabel('mu (DD)');
[r,p] = corr(B_rdm(:,1), B_dd(:,1));
title(sprintf('Luce Cohort 2 | r=%.2f, p=%.3f', r, p));
box off; axis square; refline(1,0);