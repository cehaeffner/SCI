function result = fitmodel_joint_softmax(rdm_data, dd_data)
% Jointly fits RDM and DD tasks with shared mu
% rdm_data: matrix with cols 3=cert, 4=gain, 5=pgain, 6=choice
% dd_data:  matrix with cols 2=sooner amt, 3=sooner delay, 4=later amt,
%           5=later delay, 6=choice, 7=alpha

result           = struct;
result.rdm_data  = rdm_data;
result.dd_data   = dd_data;
result.betalabel = {'mu','alpha','kappa'};
result.inx       = [1    0.8   0.15 ];
result.lb        = [0.01 0.3   0.001];
result.ub        = [20   1.6   0.5  ];
result.options   = optimset('Display','off','MaxIter',100000,'TolFun',1e-10,'TolX',1e-10,...
    'DiffMaxChange',1e-2,'DiffMinChange',1e-4,'MaxFunEvals',10000,'LargeScale','off');
warning off;

try
    [b, ~, exitflag, output, ~, ~, H] = fmincon(@setup_joint_model, result.inx, [], [], [], [], ...
        result.lb, result.ub, [], result.options, result);

    [loglike, rdm_utildiff, rdm_probchoice, dd_utildiff, dd_probchoice] = setup_joint_model(b, result);

    result.b             = b;
    result.se            = transpose(sqrt(diag(inv(H))));
    result.modelLL       = -loglike;
    result.exitflag      = exitflag;
    result.output        = output;
    result.rdm_utildiff  = rdm_utildiff;
    result.rdm_probchoice = rdm_probchoice;
    result.dd_utildiff   = dd_utildiff;
    result.dd_probchoice = dd_probchoice;

    % Pseudo-R2 against null model pooling both tasks
    choice_rdm  = rdm_data(:,6);
    choice_dd   = dd_data(:,6);
    all_choices = [choice_rdm; choice_dd];
    p_null      = mean(all_choices);
    LL_null     = sum(all_choices .* log(p_null) + (1-all_choices) .* log(1-p_null));
    result.pseudoR2 = 1 - (result.modelLL / LL_null);

catch
    fprintf(1,'joint model fit failed\n');
end
end

function [loglike, rdm_utildiff, rdm_probchoice, dd_utildiff, dd_probchoice] = setup_joint_model(x, data)
data.mu    = x(1);
data.alpha = x(2);
data.kappa = x(3);
[loglike, rdm_utildiff, rdm_probchoice, dd_utildiff, dd_probchoice] = joint_model(data);
end

function [loglike, rdm_utildiff, rdm_probchoice, dd_utildiff, dd_probchoice] = joint_model(data)

% RDM component
utilcertain  = data.rdm_data(:,3).^data.alpha;
utilgamble   = data.rdm_data(:,5) .* data.rdm_data(:,4).^data.alpha;
rdm_utildiff = utilgamble - utilcertain;
rdm_logodds  = data.mu .* rdm_utildiff;
rdm_probchoice = 1 ./ (1 + exp(-rdm_logodds));
rdm_choice   = data.rdm_data(:,6);

% DD component
alpha_dd    = data.dd_data(:,7);  % per-subject alpha from RDM (fixed)
utilsoon    = data.dd_data(:,2).^alpha_dd ./ (1 + data.kappa .* data.dd_data(:,3));
utillate    = data.dd_data(:,4).^alpha_dd ./ (1 + data.kappa .* data.dd_data(:,5));
dd_utildiff = utillate - utilsoon;
dd_logodds  = data.mu .* dd_utildiff;
dd_probchoice = 1 ./ (1 + exp(-dd_logodds));
dd_choice   = data.dd_data(:,6);

% Combine log likelihoods
rdm_probchoice(rdm_probchoice==0) = eps;
rdm_probchoice(rdm_probchoice==1) = 1-eps;
dd_probchoice(dd_probchoice==0)   = eps;
dd_probchoice(dd_probchoice==1)   = 1-eps;

LL_rdm  = -(transpose(rdm_choice(:))*log(rdm_probchoice(:)) + transpose(1-rdm_choice(:))*log(1-rdm_probchoice(:)));
LL_dd   = -(transpose(dd_choice(:))*log(dd_probchoice(:))   + transpose(1-dd_choice(:))*log(1-dd_probchoice(:)));
loglike = LL_rdm + LL_dd;
end