function [ t,f1,f2 ] = loclin( X,Y,a,b )
% The function performs the 1-d covariate ridged local linear estimator for E(Y|X) and
% the local quadratic ridged estimator for E'(Y|X).  
% Input:
% X: n*1 covariate vector;
% Y: n*1 outcome vector;
% [a,b]: the interval for which we estimate E(Y|X), [q_0.025 and q_0.975] as default.
% Output:
% t: the vector where we evaluate the estimator;
% f1: the vector of estimated E(Y|X=x);
% f2: the vector of estimated E'(Y|X=x).

if nargin < 3
    a = quantile(X,0.025);
    b = quantile(X,0.975);
end

if isrow(X)
    X = X';
end

if isrow(Y)
    Y = Y';
end

if length(unique(X)) < 4
    error('Too few data points.')
end

t = linspace(a,b,200)'; %Where to evaluate regression

f1 = zeros(length(t),1);
f2 = zeros(length(t),1);

%Bandwidth selection
opt = optimset('Display','off','MaxIter',20);
Wei = X>=a & X<=b;%Weighting function
      
ht0 = @(h) CVloss(X,Y,h,a,b);
hCV = fminbnd(ht0,(b-a).*length(unique(X))^(-0.5),(b-a)./2,opt);
fpilot = zeros(length(X),1);
WV = zeros(length(X),length(X));

for i = 1:length(X)
    if X(i) >= a && X(i) <=b
    a0 = normpdf((X-X(i))./hCV)./hCV;
    a1 = (X-X(i)).*a0;
    a2 = (X-X(i)).*a1;
    
    S0 = sum(a0);
    S1 = sum(a1);
    S2 = sum(a2);
    T0 = sum(a0.*Y);
    T1 = sum(a1.*Y);    
        
    fpilot(i) = (T0*S2-T1*S1)/(S0*S2-S1^2);    
    WV(i,:) = (a0'.*S2-a1'.*S1)./(S0*S2-S1^2);    
    end
end

nu = length(X) - 2.*trace(WV)+sum(sum(WV.^2));   
Var = sum((Y-fpilot).^2.*Wei)./nu;
       
%\hat{m}_qua
a_0 = ones(length(X),1);
a_1 = X.*a_0;
a_2 = X.*a_1;
a_3 = X.*a_2;
a_4 = X.*a_3;
a_5 = X.*a_4;
a_6 = X.*a_5;
a_7 = X.*a_6;
a_8 = X.*a_7;
b_0 = Y.*a_0;
b_1 = X.*b_0;
b_2 = X.*b_1;
b_3 = X.*b_2;
b_4 = X.*b_3;

S0 = sum(a_0); S1 = sum(a_1); S2 = sum(a_2); S3 = sum(a_3); S4 = sum(a_4);
S5 = sum(a_5); S6 = sum(a_6); S7 = sum(a_7); S8 = sum(a_8);
T0 = sum(b_0); T1 = sum(b_1); T2 = sum(b_2); T3 = sum(b_3); T4 = sum(b_4);

beta_2 = (- T4*S1^2*S5*S7 + S8*T3*S1^2*S5 + T4*S1^2*S6^2 - T3*S1^2*S6*S7 - S8*T2*S1^2*S6 + T2*S1^2*S7^2 + T4*S1*S2*S4*S7 - S8*T3*S1*S2*S4 - T4*S1*S2*S5*S6 + T3*S1*S2*S5*S7 + S8*T1*S1*S2*S6 - T1*S1*S2*S7^2 + T4*S1*S3^2*S7 - S8*T3*S1*S3^2 - 3*T4*S1*S3*S4*S6 + T3*S1*S3*S4*S7 + 2*S8*T2*S1*S3*S4 + T4*S1*S3*S5^2 + T3*S1*S3*S5*S6 - 2*T2*S1*S3*S5*S7 - S8*T1*S1*S3*S5 + T1*S1*S3*S6*S7 + S8*T0*S1*S3*S6 - T0*S1*S3*S7^2 + T4*S1*S4^2*S5 + T3*S1*S4^2*S6 - 2*T2*S1*S4^2*S7 - 2*T3*S1*S4*S5^2 + 2*T2*S1*S4*S5*S6 + T1*S1*S4*S5*S7 - S8*T0*S1*S4*S5 - T1*S1*S4*S6^2 + T0*S1*S4*S6*S7 + T0*S1*S5^2*S7 - T0*S1*S5*S6^2 - T4*S2^2*S3*S7 + S8*T3*S2^2*S3 + T4*S2^2*S4*S6 - T3*S2^2*S4*S7 - S8*T0*S2^2*S6 + T0*S2^2*S7^2 + T4*S2*S3^2*S6 - S8*T2*S2*S3^2 - T3*S2*S3*S4*S6 + 2*T2*S2*S3*S4*S7 - S8*T1*S2*S3*S4 - T3*S2*S3*S5^2 + T1*S2*S3*S5*S7 + S8*T0*S2*S3*S5 - T0*S2*S3*S6*S7 - T4*S2*S4^3 + 2*T3*S2*S4^2*S5 - T2*S2*S4^2*S6 + T1*S2*S4^2*S7 + S8*T0*S2*S4^2 - T1*S2*S4*S5*S6 - 3*T0*S2*S4*S5*S7 +...
          T0*S2*S4*S6^2 + T0*S2*S5^2*S6 + S0*T4*S2*S5*S7 - S0*S8*T3*S2*S5 - S0*T4*S2*S6^2 + S0*T3*S2*S6*S7 + S0*S8*T2*S2*S6 - S0*T2*S2*S7^2 - T4*S3^3*S5 + S8*T1*S3^3 + T4*S3^2*S4^2 + T3*S3^2*S4*S5 - 2*T1*S3^2*S4*S7 - S8*T0*S3^2*S4 + T2*S3^2*S5^2 - T1*S3^2*S5*S6 + T0*S3^2*S5*S7 - T3*S3*S4^3 - 2*T2*S3*S4^2*S5 + 2*T1*S3*S4^2*S6 + T0*S3*S4^2*S7 + T1*S3*S4*S5^2 - S0*T4*S3*S4*S7 + S0*S8*T3*S3*S4 - T0*S3*S5^3 + S0*T4*S3*S5*S6 - S0*T3*S3*S5*S7 - S0*S8*T1*S3*S6 + S0*T1*S3*S7^2 + T2*S4^4 - T1*S4^3*S5 - T0*S4^3*S6 + T0*S4^2*S5^2 + S0*T4*S4^2*S6 - S0*S8*T2*S4^2 - S0*T4*S4*S5^2 - S0*T3*S4*S5*S6 + 2*S0*T2*S4*S5*S7 + S0*S8*T1*S4*S5 - S0*T1*S4*S6*S7 + S0*T3*S5^3 - S0*T2*S5^2*S6 - S0*T1*S5^2*S7 + S0*T1*S5*S6^2)/...
         (- S8*S1^2*S4*S6 + S1^2*S4*S7^2 + S8*S1^2*S5^2 - 2*S1^2*S5*S6*S7 + S1^2*S6^3 + 2*S8*S1*S2*S3*S6 - 2*S1*S2*S3*S7^2 - 2*S8*S1*S2*S4*S5 + 2*S1*S2*S4*S6*S7 + 2*S1*S2*S5^2*S7 - 2*S1*S2*S5*S6^2 - 2*S8*S1*S3^2*S5 + 2*S1*S3^2*S6*S7 + 2*S8*S1*S3*S4^2 - 4*S1*S3*S4*S6^2 + 2*S1*S3*S5^2*S6 - 2*S1*S4^3*S7 + 4*S1*S4^2*S5*S6 - 2*S1*S4*S5^3 - S8*S2^3*S6 + S2^3*S7^2 + 2*S8*S2^2*S3*S5 - 2*S2^2*S3*S6*S7 + S8*S2^2*S4^2 - 4*S2^2*S4*S5*S7 + 2*S2^2*S4*S6^2 + S2^2*S5^2*S6 - 3*S8*S2*S3^2*S4 + 2*S2*S3^2*S5*S7 + S2*S3^2*S6^2 + 4*S2*S3*S4^2*S7 - 2*S2*S3*S4*S5*S6 - 2*S2*S3*S5^3 - 3*S2*S4^3*S6 + 3*S2*S4^2*S5^2 + S0*S8*S2*S4*S6 - S0*S2*S4*S7^2 - S0*S8*S2*S5^2 + 2*S0*S2*S5*S6*S7 - S0*S2*S6^3 + S8*S3^4 - 2*S3^3*S4*S7 - 2*S3^3*S5*S6 + 3*S3^2*S4^2*S6 + 3*S3^2*S4*S5^2 - S0*S8*S3^2*S6 + S0*S3^2*S7^2 - 4*S3*S4^3*S5 + 2*S0*S8*S3*S4*S5 - 2*S0*S3*S4*S6*S7 - 2*S0*S3*S5^2*S7 + 2*S0*S3*S5*S6^2 + S4^5 - S0*S8*S4^3 + 2*S0*S4^2*S5*S7 + S0*S4^2*S6^2 - 3*S0*S4*S5^2*S6 + S0*S5^4);
beta_3 = (S4^4*T1 + S1^2*S6^2*T3 + S2^2*S5^2*T3 + S3^2*S4^2*T3 + S0*S5^3*T2 - S2*S5^3*T0 - S1*S4^3*T4 - S2*S4^3*T3 - S3*S4^3*T2 - S4^3*S5*T0 - S3^3*S4*T4 + S3^3*S8*T0 + S2^3*S7*T4 - S2^3*S8*T3 - S0*S2*S6^2*T3 + S0*S4*S6^2*T1 - S1*S3*S6^2*T1 - S1*S4*S6^2*T0 + S2*S3*S6^2*T0 - S0*S3*S5^2*T4 - S0*S4*S5^2*T3 - S0*S5^2*S6*T1 + S1*S2*S5^2*T4 - 2*S1*S4*S5^2*T2 + S1*S5^2*S6*T0 - S2*S3*S5^2*T2 + S2*S4*S5^2*T1 + 2*S3*S4*S5^2*T0 + S0*S4^2*S5*T4 - S0*S4^2*S8*T1 + 2*S1*S4^2*S5*T3 + S1*S4^2*S6*T2 - S1*S4^2*S7*T1 + S1*S4^2*S8*T0 + 2*S2*S3*S4^2*T4 + 2*S2*S4^2*S5*T2 - 2*S2*S4^2*S6*T1 + S2*S4^2*S7*T0 - 2*S3*S4^2*S5*T1 + S3*S4^2*S6*T0 + S0*S3^2*S7*T4 - S0*S3^2*S8*T3 + S1*S3^2*S6*T4 - S1*S3^2*S8*T2 + S2*S3^2*S5*T4 - S2*S3^2*S8*T1 + S3^2*S4*S5*T2 + S3^2*S4*S6*T1 - S3^2*S4*S7*T0 - 2*S3^2*S5*S6*T0 - S2^2*S3*S6*T4 + S2^2*S3*S8*T2 - 2*S2^2*S4*S5*T4 + 2*S2^2*S4*S6*T3 - S2^2*S4*S7*T2 + S2^2*S4*S8*T1 - S2^2*S5*S7*T1 + S2^2*S5*S8*T0 - S2^2*S6*S7*T0 + S1^2*S4*S7*T4 - S1^2*S4*S8*T3 -...
          S1^2*S5*S6*T4 + S1^2*S5*S8*T2 - S1^2*S6*S7*T2 - S0*S2*S4*S7*T4 + S0*S2*S4*S8*T3 + S0*S2*S5*S6*T4 - S0*S2*S5*S8*T2 + S0*S2*S6*S7*T2 - S0*S3*S4*S6*T4 + S0*S3*S4*S8*T2 + 2*S0*S3*S5*S6*T3 - S0*S3*S5*S7*T2 + S0*S3*S5*S8*T1 - S0*S3*S6*S7*T1 - S0*S4*S5*S6*T2 + S0*S4*S5*S7*T1 - 2*S1*S2*S3*S7*T4 + 2*S1*S2*S3*S8*T3 + S1*S2*S4*S6*T4 - S1*S2*S4*S8*T2 - 2*S1*S2*S5*S6*T3 + S1*S2*S5*S7*T2 - S1*S2*S5*S8*T1 + S1*S2*S6*S7*T1 - 2*S1*S3*S4*S6*T3 + S1*S3*S4*S7*T2 + S1*S3*S4*S8*T1 + S1*S3*S5*S6*T2 - S1*S3*S5*S8*T0 + S1*S3*S6*S7*T0 + S1*S4*S5*S6*T1 - S1*S4*S5*S7*T0 - 2*S2*S3*S4*S5*T3 - S2*S3*S4*S6*T2 + S2*S3*S4*S7*T1 - 2*S2*S3*S4*S8*T0 + S2*S3*S5*S6*T1 + S2*S3*S5*S7*T0)/...
         (- S8*S1^2*S4*S6 + S1^2*S4*S7^2 + S8*S1^2*S5^2 - 2*S1^2*S5*S6*S7 + S1^2*S6^3 + 2*S8*S1*S2*S3*S6 - 2*S1*S2*S3*S7^2 - 2*S8*S1*S2*S4*S5 + 2*S1*S2*S4*S6*S7 + 2*S1*S2*S5^2*S7 - 2*S1*S2*S5*S6^2 - 2*S8*S1*S3^2*S5 + 2*S1*S3^2*S6*S7 + 2*S8*S1*S3*S4^2 - 4*S1*S3*S4*S6^2 + 2*S1*S3*S5^2*S6 - 2*S1*S4^3*S7 + 4*S1*S4^2*S5*S6 - 2*S1*S4*S5^3 - S8*S2^3*S6 + S2^3*S7^2 + 2*S8*S2^2*S3*S5 - 2*S2^2*S3*S6*S7 + S8*S2^2*S4^2 - 4*S2^2*S4*S5*S7 + 2*S2^2*S4*S6^2 + S2^2*S5^2*S6 - 3*S8*S2*S3^2*S4 + 2*S2*S3^2*S5*S7 + S2*S3^2*S6^2 + 4*S2*S3*S4^2*S7 - 2*S2*S3*S4*S5*S6 - 2*S2*S3*S5^3 - 3*S2*S4^3*S6 + 3*S2*S4^2*S5^2 + S0*S8*S2*S4*S6 - S0*S2*S4*S7^2 - S0*S8*S2*S5^2 + 2*S0*S2*S5*S6*S7 - S0*S2*S6^3 + S8*S3^4 - 2*S3^3*S4*S7 - 2*S3^3*S5*S6 + 3*S3^2*S4^2*S6 + 3*S3^2*S4*S5^2 - S0*S8*S3^2*S6 + S0*S3^2*S7^2 - 4*S3*S4^3*S5 + 2*S0*S8*S3*S4*S5 - 2*S0*S3*S4*S6*S7 - 2*S0*S3*S5^2*S7 + 2*S0*S3*S5*S6^2 + S4^5 - S0*S8*S4^3 + 2*S0*S4^2*S5*S7 + S0*S4^2*S6^2 - 3*S0*S4*S5^2*S6 + S0*S5^4);
beta_4 = (- T4*S1^2*S4*S6 + S7*T3*S1^2*S4 + T4*S1^2*S5^2 - T3*S1^2*S5*S6 - S7*T2*S1^2*S5 + T2*S1^2*S6^2 + 2*T4*S1*S2*S3*S6 - 2*S7*T3*S1*S2*S3 - 2*T4*S1*S2*S4*S5 + T3*S1*S2*S4*S6 + S7*T2*S1*S2*S4 + T3*S1*S2*S5^2 - T2*S1*S2*S5*S6 + S7*T1*S1*S2*S5 - T1*S1*S2*S6^2 - 2*T4*S1*S3^2*S5 + T3*S1*S3^2*S6 + S7*T2*S1*S3^2 + 2*T4*S1*S3*S4^2 - 3*T2*S1*S3*S4*S6 - S7*T1*S1*S3*S4 + T2*S1*S3*S5^2 + T1*S1*S3*S5*S6 + S7*T0*S1*S3*S5 - T0*S1*S3*S6^2 - T3*S1*S4^3 + T2*S1*S4^2*S5 + T1*S1*S4^2*S6 - S7*T0*S1*S4^2 - T1*S1*S4*S5^2 + 2*T0*S1*S4*S5*S6 - T0*S1*S5^3 - T4*S2^3*S6 + S7*T3*S2^3 + 2*T4*S2^2*S3*S5 - T3*S2^2*S3*S6 - S7*T2*S2^2*S3 + T4*S2^2*S4^2 - 2*T3*S2^2*S4*S5 + T2*S2^2*S4*S6 - S7*T1*S2^2*S4 + T1*S2^2*S5*S6 - S7*T0*S2^2*S5 + T0*S2^2*S6^2 - 3*T4*S2*S3^2*S4 + T3*S2*S3^2*S5 + T2*S2*S3^2*S6 + S7*T1*S2*S3^2 + 2*T3*S2*S3*S4^2 + 2*S7*T0*S2*S3*S4 - 2*T1*S2*S3*S5^2 - 2*T0*S2*S3*S5*S6 - T2*S2*S4^3 + T1*S2*S4^2*S5 - 2*T0*S2*S4^2*S6 + 2*T0*S2*S4*S5^2 + S0*T4*S2*S4*S6 - S0*S7*T3*S2*S4 -...
          S0*T4*S2*S5^2 + S0*T3*S2*S5*S6 + S0*S7*T2*S2*S5 - S0*T2*S2*S6^2 + T4*S3^4 - T3*S3^3*S4 - T2*S3^3*S5 - T1*S3^3*S6 - S7*T0*S3^3 + T2*S3^2*S4^2 + 2*T1*S3^2*S4*S5 + 2*T0*S3^2*S4*S6 + T0*S3^2*S5^2 - S0*T4*S3^2*S6 + S0*S7*T3*S3^2 - T1*S3*S4^3 - 3*T0*S3*S4^2*S5 + 2*S0*T4*S3*S4*S5 - S0*T3*S3*S4*S6 - S0*S7*T2*S3*S4 - S0*T3*S3*S5^2 + S0*T2*S3*S5*S6 - S0*S7*T1*S3*S5 + S0*T1*S3*S6^2 + T0*S4^4 - S0*T4*S4^3 + S0*T3*S4^2*S5 + S0*T2*S4^2*S6 + S0*S7*T1*S4^2 - S0*T2*S4*S5^2 - 2*S0*T1*S4*S5*S6 + S0*T1*S5^3)/...
         (- S8*S1^2*S4*S6 + S1^2*S4*S7^2 + S8*S1^2*S5^2 - 2*S1^2*S5*S6*S7 + S1^2*S6^3 + 2*S8*S1*S2*S3*S6 - 2*S1*S2*S3*S7^2 - 2*S8*S1*S2*S4*S5 + 2*S1*S2*S4*S6*S7 + 2*S1*S2*S5^2*S7 - 2*S1*S2*S5*S6^2 - 2*S8*S1*S3^2*S5 + 2*S1*S3^2*S6*S7 + 2*S8*S1*S3*S4^2 - 4*S1*S3*S4*S6^2 + 2*S1*S3*S5^2*S6 - 2*S1*S4^3*S7 + 4*S1*S4^2*S5*S6 - 2*S1*S4*S5^3 - S8*S2^3*S6 + S2^3*S7^2 + 2*S8*S2^2*S3*S5 - 2*S2^2*S3*S6*S7 + S8*S2^2*S4^2 - 4*S2^2*S4*S5*S7 + 2*S2^2*S4*S6^2 + S2^2*S5^2*S6 - 3*S8*S2*S3^2*S4 + 2*S2*S3^2*S5*S7 + S2*S3^2*S6^2 + 4*S2*S3*S4^2*S7 - 2*S2*S3*S4*S5*S6 - 2*S2*S3*S5^3 - 3*S2*S4^3*S6 + 3*S2*S4^2*S5^2 + S0*S8*S2*S4*S6 - S0*S2*S4*S7^2 - S0*S8*S2*S5^2 + 2*S0*S2*S5*S6*S7 - S0*S2*S6^3 + S8*S3^4 - 2*S3^3*S4*S7 - 2*S3^3*S5*S6 + 3*S3^2*S4^2*S6 + 3*S3^2*S4*S5^2 - S0*S8*S3^2*S6 + S0*S3^2*S7^2 - 4*S3*S4^3*S5 + 2*S0*S8*S3*S4*S5 - 2*S0*S3*S4*S6*S7 - 2*S0*S3*S5^2*S7 + 2*S0*S3*S5*S6^2 + S4^5 - S0*S8*S4^3 + 2*S0*S4^2*S5*S7 + S0*S4^2*S6^2 - 3*S0*S4*S5^2*S6 + S0*S5^4);

f_qua_2 = zeros(length(X),1);
f_qua_4 = zeros(length(X),1);
for i = 1:length(X) 
    if X(i) >= a && X(i) <=b
        f_qua_2(i) = 2*beta_2 + 6*beta_3*X(i) + 12*beta_4*X(i)^2;
        f_qua_4(i) = 24*beta_4;
    end
end

Theta_24 = sum(f_qua_2.*f_qua_4.*(a<=X&X<=b))/length(X);

if Theta_24>0
    C_2K = (15/(16*sqrt(pi)))^(1/7);
else
    C_2K = (3/(8*sqrt(pi)))^(1/7);
end
hCV3 = C_2K * (Var / (length(X) * abs(Theta_24)))^(1/7); 
if Theta_24 == 0
    hCV3 = 2*(b-a);
end

fCV3 = zeros(length(X),1); 
ffCV3 = zeros(length(X),1);

for i = 1:length(X)
    if X(i) >= a && X(i) <=b      
    a0 = normpdf((X-X(i))./hCV3)./hCV3;
    a1 = (X-X(i)).*a0;
    a2 = (X-X(i)).*a1;
    a3 = (X-X(i)).*a2;
    a4 = (X-X(i)).*a3;
    a5 = (X-X(i)).*a4;
    a6 = (X-X(i)).*a5;
    
    S0 = sum(a0);
    S1 = sum(a1);
    S2 = sum(a2);
    S3 = sum(a3);
    S4 = sum(a4);
    S5 = sum(a5);
    S6 = sum(a6);
    
    T0 = sum(a0.*Y);
    T1 = sum(a1.*Y); 
    T2 = sum(a2.*Y); 
    T3 = sum(a3.*Y);
    
    fCV3(i) = 2*(T3*(- S5*S1^2 + S4*S1*S2 + S1*S3^2 - S2^2*S3 + S0*S5*S2 - S0*S4*S3)...
                -T0*(- S6*S2^2 + S5*S2*S3 + S2*S4^2 - S3^2*S4 + S1*S6*S3 - S1*S5*S4)...
                -T1*(S3^3 - S0*S3*S6 + S0*S4*S5 + S1*S2*S6 - S1*S3*S5 - S2*S3*S4)...
                +T2*(S6*S1^2 - 2*S1*S3*S4 + S2*S3^2 + S0*S4^2 - S0*S2*S6))...
               /(S6*S1^2*S4 - S1^2*S5^2 - 2*S6*S1*S2*S3 + 2*S1*S2*S4*S5 + 2*S1*S3^2*S5 -...
               2*S1*S3*S4^2 + S6*S2^3 - 2*S2^2*S3*S5 - S2^2*S4^2 + 3*S2*S3^2*S4 - S0*S6*S2*S4 + S0*S2*S5^2 - S3^4 + S0*S6*S3^2 - 2*S0*S3*S4*S5 + S0*S4^3);            
    
    ffCV3(i) = 6*(T3*(S4*S1^2 - 2*S1*S2*S3 + S2^3 - S0*S4*S2 + S0*S3^2)...
        -T0*(S5*S2^2 - 2*S2*S3*S4 + S3^3 - S1*S5*S3 + S1*S4^2)...
        +T2*(- S5*S1^2 + S4*S1*S2 + S1*S3^2 - S2^2*S3 + S0*S5*S2 - S0*S4*S3)...
        +T1*(- S2^2*S4 + S2*S3^2 + S1*S5*S2 - S1*S3*S4 - S0*S5*S3 + S0*S4^2))...
    /(S6*S1^2*S4 - S1^2*S5^2 - 2*S6*S1*S2*S3 + 2*S1*S2*S4*S5 + 2*S1*S3^2*S5 - 2*S1*S3*S4^2 +...
    S6*S2^3 - 2*S2^2*S3*S5 - S2^2*S4^2 + 3*S2*S3^2*S4 - S0*S6*S2*S4 + S0*S2*S5^2 - S3^4 + S0*S6*S3^2 - 2*S0*S3*S4*S5 + S0*S4^3);
    end
end

Theta1 = sum(fCV3.^2.*Wei)/length(X);
h1 = (Var./(2.*sqrt(pi).*Theta1.*length(X))).^(1/5);
if Theta1 == 0
    h1 = 2*(b-a);
end

Theta2 = sum(ffCV3.^2.*Wei)/length(X);
h2 = 0.8843.*(Var./(Theta2.*length(X))).^(1/7);
if Theta2 == 0
    h2 = 2*(b-a);
end

delta = length(unique(X))^(-2);
for i = 1:length(t)
    a0 = normpdf((X-t(i))./h1)./h1;
    a1 = (X-t(i)).*a0;
    a2 = (X-t(i)).*a1;
    
    S0 = sum(a0);
    S1 = sum(a1);
    S2 = sum(a2);
    T0 = sum(a0.*Y);
    T1 = sum(a1.*Y);    
    den = S0*S2-S1^2;
    if den>0 && den<delta
        den = den+delta;
    elseif den<0 && den>-delta
        den = den-delta;
    end
    f1(i) = (T0*S2-T1*S1)/den;
    
    a0 = normpdf((X-t(i))./h2)./h2;
    a1 = (X-t(i)).*a0;
    a2 = (X-t(i)).*a1;
    a3 = (X-t(i)).*a2;
    a4 = (X-t(i)).*a3;
    
    S0 = sum(a0);
    S1 = sum(a1);
    S2 = sum(a2);
    S3 = sum(a3);
    S4 = sum(a4);  
    T0 = sum(a0.*Y);
    T1 = sum(a1.*Y);
    T2 = sum(a2.*Y);   
    den = S4*S1^2 - 2*S1*S2*S3 + S2^3 - S0*S4*S2 + S0*S3^2;   
    if den>0 && den<delta
        den = den+delta;
    elseif den<0 && den>-delta
        den = den-delta;
    end
    f2(i) = (T2*(S0*S3 - S1*S2)+T0*(S1*S4 - S2*S3)-T1*(- S2^2 + S0*S4))/den;
end

end

