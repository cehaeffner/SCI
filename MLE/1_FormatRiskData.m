% Format for matlab MLE fit

clear;clc;

%% Cohort 1 risky decision-making (RDM) data
% Load JSON
raw = jsondecode(fileread('rdm_1.json'));

ns     = raw.ns;
Tsubj  = raw.Tsubj;

% Build cumulative trial index
trial_ends   = cumsum(Tsubj);
trial_starts = [1; trial_ends(1:end-1) + 1];

% Build per-subject matrices
% Build per-subject matrices
alldata = struct();
for s = 1:ns
    idx = trial_starts(s):trial_ends(s);
    
    mat = zeros(length(idx), 6);
    mat(:,3) = raw.cert(idx);
    mat(:,4) = raw.gain(idx);
    mat(:,5) = raw.pgain(idx);
    mat(:,6) = raw.gamble(idx);
    
    alldata(s).subid = raw.subid(idx(1));  % take the first trial's subid for that subject
    alldata(s).data  = mat;
end

save('rdm_1.mat', 'alldata');

% RDM bounds - exploring what the theoretical bounds are
alpha_eq = [];
for s = 1:ns
    idx = trial_starts(s):trial_ends(s);
    cert  = raw.cert(idx);
    gain  = raw.gain(idx);
    pgain = raw.pgain(idx);
    
    % solve p * gain^alpha = cert^alpha for alpha
    % alpha = log(p) / log(cert/gain)
    denom = log(cert ./ gain);
    numer = log(pgain);
    
    % exclude trials where denom is zero or near-zero (cert == gain)
    valid = abs(denom) > 1e-6;
    alpha_eq_s = numer(valid) ./ denom(valid);
    alpha_eq   = [alpha_eq; alpha_eq_s];
end

% exclude imaginary, inf, or nan values
alpha_eq = alpha_eq(isfinite(alpha_eq) & isreal(alpha_eq));

fprintf('Alpha bounds from indifference:\n');
fprintf('  Min: %.4f\n', min(alpha_eq));
fprintf('  Max: %.4f\n', max(alpha_eq));
fprintf('  Suggested lb: %.4f, ub: %.4f\n', floor(min(alpha_eq)*10)/10, ceil(max(alpha_eq)*10)/10);

%% Cohort 2 risky decision-making (RDM) data
% Load JSON
raw = jsondecode(fileread('rdm_2.json'));

ns     = raw.ns;
Tsubj  = raw.Tsubj;

% Build cumulative trial index
trial_ends   = cumsum(Tsubj);
trial_starts = [1; trial_ends(1:end-1) + 1];

% Build per-subject matrices
% Build per-subject matrices
alldata = struct();
for s = 1:ns
    idx = trial_starts(s):trial_ends(s);
    
    mat = zeros(length(idx), 6);
    mat(:,3) = raw.cert(idx);
    mat(:,4) = raw.gain(idx);
    mat(:,5) = raw.pgain(idx);
    mat(:,6) = raw.gamble(idx);
    
    alldata(s).subid = raw.subid(idx(1));  % take the first trial's subid for that subject
    alldata(s).data  = mat;
end

save('rdm_2.mat', 'alldata');

% RDM bounds - exploring what the theoretical bounds are
alpha_eq = [];
for s = 1:ns
    idx = trial_starts(s):trial_ends(s);
    cert  = raw.cert(idx);
    gain  = raw.gain(idx);
    pgain = raw.pgain(idx);
    
    % solve p * gain^alpha = cert^alpha for alpha
    % alpha = log(p) / log(cert/gain)
    denom = log(cert ./ gain);
    numer = log(pgain);
    
    % exclude trials where denom is zero or near-zero (cert == gain)
    valid = abs(denom) > 1e-6;
    alpha_eq_s = numer(valid) ./ denom(valid);
    alpha_eq   = [alpha_eq; alpha_eq_s];
end

% exclude imaginary, inf, or nan values
alpha_eq = alpha_eq(isfinite(alpha_eq) & isreal(alpha_eq));

fprintf('Alpha bounds from indifference:\n');
fprintf('  Min: %.4f\n', min(alpha_eq));
fprintf('  Max: %.4f\n', max(alpha_eq));
fprintf('  Suggested lb: %.4f, ub: %.4f\n', floor(min(alpha_eq)*10)/10, ceil(max(alpha_eq)*10)/10);
