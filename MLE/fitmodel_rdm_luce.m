function result = fitmodel_rdm_luce(indata)

% RESULT = fitmodel_rdm_luce(INDATA)
%
% INDATA is a matrix with at least 6 columns (col 3 certain amount, col 4
% win amount, col 5 is win probability, col 6 chose risky is 1, chose safe is 0)

result           = struct;
result.data      = indata;
result.betalabel = {'mu','alpha'}; 
result.inx       = [1     0.8];   %initial values for parameters
result.lb        = [0.01  0.1]; %min values possible for design matrix
result.ub        = [20    1.8];   %max values
result.options   = optimset('Display','off','MaxIter',100000,'TolFun',1e-10,'TolX',1e-10,...
    'DiffMaxChange',1e-2,'DiffMinChange',1e-4,'MaxFunEvals',10000,'LargeScale','off');
warning off;                    %to see outputs use 'Display','iter'

try
    [b, ~, exitflag, output, ~, ~, H] = fmincon(@setup_rdm_luce_model,result.inx,[],[],[],[],result.lb,result.ub,[],result.options,result);
    clear temp;
    [loglike, utildiff, logodds, probchoice] = setup_rdm_luce_model(b, result);
    result.b          = b;      %parameter estimates
    result.se         = transpose(sqrt(diag(inv(H)))); %SEs for parameters from inverse of the Hessian
    result.modelLL    = -loglike;
    result.exitflag   = exitflag;
    result.output     = output;
    result.utildiff   = utildiff;
    result.logodds    = logodds;
    result.probchoice = probchoice;

    % adding pseudo-R2
    p_null = mean(indata(:,6));
    LL_null = sum(indata(:,6) .* log(p_null) + (1-indata(:,6)) .* log(1-p_null));
    LL_model = result.modelLL;
    result.pseudoR2 = 1 - (LL_model / LL_null); 

catch
    fprintf(1,'model fit failed\n');
end

end


function [loglike, utildiff, logodds, probchoice] = setup_rdm_luce_model(x, data)

data.mu     = x(1);
data.alpha  = x(2);

[loglike, utildiff, logodds, probchoice] = rdm_luce_model(data);

end


function [loglike, utildiff, logodds, probchoice] = rdm_luce_model(data)

%data.data is a matrix with at least 6 columns (col 3 certain amount, col 4
%win amount, col 5 is win probability, col 6 chose risky is 1, chose safe is 0)
%data.mu is inverse temperature and data.alpha is risk aversion
%parameters. function returns -loglikelihood and vectors for trial-by-trial
%utility difference, logodds, and probability of taking the risky option

utilcertain   = data.data(:,3).^data.alpha;                 %utility for certain option (probability is 1)
utilgamble    = data.data(:,5).*data.data(:,4).^data.alpha; %utility for risky option
probchoice    = utilgamble.^data.mu ./ (utilgamble.^data.mu + utilcertain.^data.mu);
utildiff = nan(size(probchoice)); 
logodds = nan(size(probchoice));                     % not naturally computed in luce model as in softmax
choice        = data.data(:,6);                      %1 chose risky, 0 chose safe

probchoice(probchoice==0) = eps;                     %to prevent fmincon crashing from log zero
probchoice(probchoice==1) = 1-eps;
loglike       = - (transpose(choice(:))*log(probchoice(:)) + transpose(1-choice(:))*log(1-probchoice(:)));
loglike       = sum(loglike);                       %number to minimize

end