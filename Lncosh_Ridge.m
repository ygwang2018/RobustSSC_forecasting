function [Optimal_B,Estimated_Kappa, Optimal_lambda]=Lncosh_Ridge(X_Predictors,Y_Response)
[n p]=size(X_Predictors);
BasicSVR = fitrsvm(X_Predictors,Y_Response,'KernelFunction','linear','Epsilon',0,'Standardize',true);
SVR_Fitted=predict(BasicSVR,X_Predictors);
Estimated_Residual=Y_Response-SVR_Fitted;
%Parameter estimation
x0=1;
lb=0.01;
ub=[];
A = [];
b = [];
Aeq = [];
beq = [];
Estimated_Kappa=fmincon(@(kappa)ParameterEstimation(Estimated_Residual,kappa),x0,A,b,Aeq,beq,lb,ub);
%Beta estimation
nvars=p+1;
x0=ones(1,nvars);
lb=[];
ub=[];
%%
DATA=[X_Predictors,Y_Response];
indices = crossvalind('Kfold',n,5);
t=1;
for lambda_sequence=10.^([-100:40]/20)
SubMSE=[];
for k=1:5
    test = (indices == k); 
    train = ~test;
    Sub_X_Predictors_Train= DATA(train,1:end-1);
    Sub_Y_Response_Train=DATA(train,end);
    Sub_X_Predictors_Test=DATA(test,1:end-1);
    Sub_Y_Response_Test=DATA(test,end);
    %options = optimoptions('fmincon','Algorithm','active-set');
    Optimal_B_Try = fmincon(@(B)lncoshl2(B,Sub_X_Predictors_Train,Sub_Y_Response_Train,Estimated_Kappa,lambda_sequence),x0,A,b,Aeq,beq,lb,ub);
    [nt pt]=size(Sub_X_Predictors_Test);
    Sub_Y_Preds_Test=[Sub_X_Predictors_Test,ones(nt,1)]*Optimal_B_Try';
    SubMSE(k)=mean((Sub_Y_Preds_Test-Sub_Y_Response_Test).^2);
end
CV_MSE(t)=mean(SubMSE);
t=t+1;
end
[~,index] = sort(CV_MSE);
Optimal_Index=index(1);
lambda_sequence=10.^([-120:40]/20);
Optimal_lambda=lambda_sequence(Optimal_Index);
Optimal_B = fmincon(@(B)lncoshl2(B,X_Predictors,Y_Response,Estimated_Kappa,Optimal_lambda),x0,A,b,Aeq,beq,lb,ub);
end

