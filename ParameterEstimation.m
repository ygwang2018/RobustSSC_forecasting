function loss=ParameterEstimation(Residuals,kappa)
n=length(Residuals);
a1=n*log(pi*kappa);
a2=[];
for i=1:n
a2(i)=log((exp(Residuals(i)/kappa)+exp(-Residuals(i)/kappa))/2);
end
loss=a1+sum(a2);
end