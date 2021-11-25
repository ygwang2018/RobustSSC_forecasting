function Loss=lncoshl2(B,Predictors,Response,Kappa,lambda)
n=length(Response);
a1=[];
for i=1:n
    y_hat(i)=[Predictors(i,:),1]*B';
    r(i)=Response(i)- y_hat(i);
    a1(i)=log((exp(r(i)/Kappa)+exp(-r(i)/Kappa))/2);
end
a2=sum(B(1:end-1).^2);
Loss=mean(a1)/2+lambda*a2;
end