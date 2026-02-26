function result = fitmodel_ddnlh_luce(indata)

% RESULT = fitmodel_itc(INDATA)
%
% INDATA is a matrix with at least 6 columns (col 2 sooner amount, col 3
% sooner delay length (# of days), col 4 later amount, col 5 later length
% (# of days), col 6 chose later is 1, chose sooner is 0)

result           = struct;
result.data      = indata;
result.betalabel = {'mu','kappa'}; 
result.inx       = [1     0.15 ];   %initial values for parameters
result.lb        = [0.01  0.001];   %min values possible for design matrix
result.ub        = [20    0.5  ];   %max values
result.options   = optimset('Display','off','MaxIter',100000,'TolFun',1e-10,'TolX',1e-10,...
    'DiffMaxChange',1e-2,'DiffMinChange',1e-4,'MaxFunEvals',10000,'LargeScale','off');
warning off;                    %to see outputs use 'Display','iter'


[b, ~, exitflag, output, ~, ~, H] = fmincon(@setup_ddnlh_model,result.inx,[],[],[],[],result.lb,result.ub,[],result.options,result);
clear temp;
[loglike, utildiff, logodds, probchoice] = setup_ddnlh_model(b, result);
result.b          = b;
result.se         = transpose(sqrt(diag(inv(H))));
result.modelLL    = -loglike;
result.exitflag   = exitflag;
result.output     = output;
result.utildiff   = utildiff;
result.logodds    = logodds;
result.probchoice = probchoice;
p_null = mean(indata(:,6));
LL_null = sum(indata(:,6) .* log(p_null) + (1-indata(:,6)) .* log(1-p_null));
LL_model = result.modelLL;
result.pseudoR2 = 1 - (LL_model / LL_null);
end


function [loglike, utildiff, logodds, probchoice] = setup_ddnlh_model(x, data)

data.mu         = x(1);
data.kappa      = x(2);

[loglike, utildiff, logodds, probchoice] = ddnlh_model(data);

end


function [loglike, utildiff, logodds, probchoice] = ddnlh_model(data)

%data.data is a matrix with at least 6 columns (col 2 sooner amount, 
%col 3 sooner delay length, col 4 later amount, col 5 delay length, 
%col 6 chose later is 1, chose sooner is 0), col 7 is alpha parameter.
%data.kappa and data.mu are discount factor and inverse temperature
%parameters. function returns -loglikelihood and vectors for trial-by-trial
%utility difference, logodds, and probability of taking the later option

utilsoon     = (data.data(:,2)).^(data.data(:,7)) ./ (1 + data.kappa.*data.data(:,3)); %utility for sooner option
utillate     = (data.data(:,4)).^(data.data(:,7)) ./ (1 + data.kappa.*data.data(:,5)); % utility for later option
probchoice   = utillate.^data.mu ./ (utillate.^data.mu + utilsoon.^data.mu);
utildiff = nan(size(probchoice)); 
logodds      = nan(size(probchoice));              % not naturally computed in luce model as in softmax
choice       = data.data(:,6);                     %1 chose later, 0 chose soner

probchoice(probchoice==0) = eps;                  %to prevent fmincon crashing from log zero
probchoice(probchoice==1) = 1-eps;
loglike       = - (transpose(choice(:))*log(probchoice(:)) + transpose(1-choice(:))*log(1-probchoice(:)));
loglike       = sum(loglike);                     %number to minimize

end