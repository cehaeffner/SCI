% Format for matlab MLE fit
clear;clc;

%% Cohort 1
rdm_versions = {'fitrdmdata_luce_1.mat', 'fitrdmdata_softmax_1.mat'};
save_names   = {'dd_luce_1.mat', 'dd_softmax_1.mat'};
rdm_fields   = {'rdm_1', 'rdm_1'}; % adjust if variable names differ inside each .mat

for v = 1:2
    % Load JSON
    raw = jsondecode(fileread('dd_1.json'));
    ns          = raw.ns;
    Tsubj       = raw.Tsubj;
    trial_ends  = cumsum(Tsubj);
    trial_starts = [1; trial_ends(1:end-1) + 1];

    % Load RDM alphas for this version
    rdm_tmp    = load(rdm_versions{v});
    rdm_data   = rdm_tmp.(rdm_fields{v});
    rdm_alphas = vertcat(rdm_data.b_rdm);
    rdm_alphas = rdm_alphas(:,2);
    rdm_subids = [rdm_data.subid]';

    alldata = struct();
    for s = 1:ns
        idx     = trial_starts(s):trial_ends(s);
        subid_s = raw.subid(idx(1));
        rdm_idx = find(rdm_subids == subid_s);

        mat = zeros(length(idx), 7);
        mat(:,2) = raw.amount_sooner(idx);
        mat(:,3) = 0;
        mat(:,4) = raw.amount_later(idx);
        mat(:,5) = raw.delay_later(idx);
        mat(:,6) = raw.choice(idx);
        mat(:,7) = rdm_alphas(rdm_idx);

        alldata(s).subid = subid_s;
        alldata(s).data  = mat;
    end

    save(save_names{v}, 'alldata');
    fprintf('Saved %s\n', save_names{v});
end

%% Cohort 2
rdm_versions = {'fitrdmdata_luce_2.mat', 'fitrdmdata_softmax_2.mat'};
save_names   = {'dd_luce_2.mat', 'dd_softmax_2.mat'};
rdm_fields   = {'rdm_2', 'rdm_2'}; % adjust if variable names differ inside each .mat

for v = 1:2
    % Load JSON
    raw = jsondecode(fileread('dd_2.json'));
    ns          = raw.ns;
    Tsubj       = raw.Tsubj;
    trial_ends  = cumsum(Tsubj);
    trial_starts = [1; trial_ends(1:end-1) + 1];

    % Load RDM alphas for this version
    rdm_tmp    = load(rdm_versions{v});
    rdm_data   = rdm_tmp.(rdm_fields{v});
    rdm_alphas = vertcat(rdm_data.b_rdm);
    rdm_alphas = rdm_alphas(:,2);
    rdm_subids = [rdm_data.subid]';

    alldata = struct();
    for s = 1:ns
        idx     = trial_starts(s):trial_ends(s);
        subid_s = raw.subid(idx(1));
        rdm_idx = find(rdm_subids == subid_s);

        mat = zeros(length(idx), 7);
        mat(:,2) = raw.amount_sooner(idx);
        mat(:,3) = 0;
        mat(:,4) = raw.amount_later(idx);
        mat(:,5) = raw.delay_later(idx);
        mat(:,6) = raw.choice(idx);
        mat(:,7) = rdm_alphas(rdm_idx);

        alldata(s).subid = subid_s;
        alldata(s).data  = mat;
    end

    save(save_names{v}, 'alldata');
    fprintf('Saved %s\n', save_names{v});
end
