function [ S ] = CVloss( X,Y,h,a,b )
%Cross-validation loss function using the local linear estimator.
S = 0;

for i = 1:length(X)
    if X(i) >= a && X(i) <=b
    Xi = X;
    Xi(i) = [];
    Yi = Y;
    Yi(i) = [];
    
    a0 = normpdf((Xi-X(i))./h)./h;
    a1 = (Xi-X(i)).*a0;
    a2 = (Xi-X(i)).*a1;
    
    S0 = sum(a0);
    S1 = sum(a1);
    S2 = sum(a2);
    T0 = sum(a0.*Yi);
    T1 = sum(a1.*Yi);    
    
    Yi_hat = (T0*S2-T1*S1)/(S0*S2-S1^2);
    S = S + (Y(i)-Yi_hat).^2;
    end
end
  
end

