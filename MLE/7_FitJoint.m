% Fit rdm and dd together with shared mu
clear;clc;

%% Cohort 1 Softmax
rdm_1 = load('rdm_1.mat'); rdm_1 = rdm_1.alldata;
dd_1 = load('dd_softmax_1.mat'); dd_1 = dd_1.alldata;
for s = 1:length(rdm_1)
    dd_subids = [dd_1.subid]';
    dd_idx    = find(dd_subids == rdm_1(s).subid);
    if ~isempty(dd_idx)
        t_rdm  = rdm_1(s).data;
        t_dd   = dd_1(dd_idx).data;
        result_joint          = fitmodel_joint_softmax(t_rdm, t_dd);
        rdm_1(s).result_joint = result_joint;
        rdm_1(s).b_joint      = result_joint.b;
        rdm_1(s).pr2_joint    = result_joint.pseudoR2;
    else
        warning('No DD match for subid %d', rdm_1(s).subid);
    end
end
save('fitjointdata_softmax_1.mat', 'rdm_1');

%% Cohort 1 Luce
rdm_1 = load('rdm_1.mat'); rdm_1 = rdm_1.alldata;
dd_1 = load('dd_luce_1.mat'); dd_1 = dd_1.alldata;
for s = 1:length(rdm_1)
    dd_subids = [dd_1.subid]';
    dd_idx    = find(dd_subids == rdm_1(s).subid);
    if ~isempty(dd_idx)
        t_rdm  = rdm_1(s).data;
        t_dd   = dd_1(dd_idx).data;
        result_joint          = fitmodel_joint_luce(t_rdm, t_dd);
        rdm_1(s).result_joint = result_joint;
        rdm_1(s).b_joint      = result_joint.b;
        rdm_1(s).pr2_joint    = result_joint.pseudoR2;
    else
        warning('No DD match for subid %d', rdm_1(s).subid);
    end
end
save('fitjointdata_luce_1.mat', 'rdm_1');

%% Cohort 2 Softmax
rdm_2 = load('rdm_2.mat'); rdm_2 = rdm_2.alldata;
dd_2 = load('dd_softmax_2.mat'); dd_2 = dd_2.alldata;
for s = 1:length(rdm_2)
    dd_subids = [dd_2.subid]';
    dd_idx    = find(dd_subids == rdm_2(s).subid);
    if ~isempty(dd_idx)
        t_rdm  = rdm_2(s).data;
        t_dd   = dd_2(dd_idx).data;
        result_joint          = fitmodel_joint_softmax(t_rdm, t_dd);
        rdm_2(s).result_joint = result_joint;
        rdm_2(s).b_joint      = result_joint.b;
        rdm_2(s).pr2_joint    = result_joint.pseudoR2;
    else
        warning('No DD match for subid %d', rdm_2(s).subid);
    end
end
save('fitjointdata_softmax_2.mat', 'rdm_2');

%% Cohort 2 Luce
rdm_2 = load('rdm_2.mat'); rdm_2 = rdm_2.alldata;
dd_2 = load('dd_luce_2.mat'); dd_2 = dd_2.alldata;
for s = 1:length(rdm_2)
    dd_subids = [dd_2.subid]';
    dd_idx    = find(dd_subids == rdm_2(s).subid);
    if ~isempty(dd_idx)
        t_rdm  = rdm_2(s).data;
        t_dd   = dd_2(dd_idx).data;
        result_joint          = fitmodel_joint_luce(t_rdm, t_dd);
        rdm_2(s).result_joint = result_joint;
        rdm_2(s).b_joint      = result_joint.b;
        rdm_2(s).pr2_joint    = result_joint.pseudoR2;
    else
        warning('No DD match for subid %d', rdm_2(s).subid);
    end
end
save('fitjointdata_luce_2.mat', 'rdm_2');

%% Cohort 3 Softmax
rdm_3 = load('rdm_3.mat'); rdm_3 = rdm_3.alldata;
dd_3 = load('dd_softmax_3.mat'); dd_3 = dd_3.alldata;
for s = 1:length(rdm_3)
    dd_subids = [dd_3.subid]';
    dd_idx    = find(dd_subids == rdm_3(s).subid);
    if ~isempty(dd_idx)
        t_rdm  = rdm_3(s).data;
        t_dd   = dd_3(dd_idx).data;
        result_joint          = fitmodel_joint_softmax(t_rdm, t_dd);
        rdm_3(s).result_joint = result_joint;
        rdm_3(s).b_joint      = result_joint.b;
        rdm_3(s).pr2_joint    = result_joint.pseudoR2;
    else
        warning('No DD match for subid %d', rdm_3(s).subid);
    end
end
save('fitjointdata_softmax_3.mat', 'rdm_3');

%% Cohort 3 Luce
rdm_3 = load('rdm_3.mat'); rdm_3 = rdm_3.alldata;
dd_3 = load('dd_luce_3.mat'); dd_3 = dd_3.alldata;
for s = 1:length(rdm_3)
    dd_subids = [dd_3.subid]';
    dd_idx    = find(dd_subids == rdm_3(s).subid);
    if ~isempty(dd_idx)
        t_rdm  = rdm_3(s).data;
        t_dd   = dd_3(dd_idx).data;
        result_joint          = fitmodel_joint_luce(t_rdm, t_dd);
        rdm_3(s).result_joint = result_joint;
        rdm_3(s).b_joint      = result_joint.b;
        rdm_3(s).pr2_joint    = result_joint.pseudoR2;
    else
        warning('No DD match for subid %d', rdm_3(s).subid);
    end
end
save('fitjointdata_luce_3.mat', 'rdm_3');