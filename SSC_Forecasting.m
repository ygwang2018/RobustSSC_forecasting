clc,clear
load data
%% Data preprocessing
for t=1:1890
    SSC_Average(t)=mean(SSC(6*(t-1)+1:6*t));
end
SSC=SSC_Average;
%%
figure(22)
subplot(2,1,1)
plot(SSC(1:1680))
subplot(2,1,2)
plot(SSC(1676:end))

%% The original data exploration
% Results: ARIMA(1,1,3) ar1 0.6362, ma1 -0.6718, ma2 -0.2484, ma3 0.0784 
%% Empirical mode decomposition (EMD)
[imf,residual,info] = emd(SSC);
SubSSC=[imf,residual];
figure(1)
for i=1:7
    subplot(7,1,i)
    plot(SubSSC(:,i),'k')
end
csvwrite('SubSSC.csv',SubSSC)
%% Firstly, the arima model is employed to show the present of the outliers and the pattern (linear or nonlinear) systems
%% Here, the basic modelling is implenmented by function "auto.arima" in R Package "forecast"
%% Results are recorded as:
% SubSSC1: ARIMA(2,0,2) ar1 0.6344 ar2 -0.2029 ma1 -0.3291 ma2 -0.2350 mean -0.0167
% SubSSC2: ARIMA(4,0,2) ar1 2.4369 ar2 -2.6509 ar3 1.5008 ar4 -0.4159 ma1 0.6425 ma2 0.1571
% SubSSC3: ARIMA(1,0,0) ar1 0.9866
% SubSSC4: ARIMA(0,0,0) mean -0.0126
% SubSSC5: ARIMA(0,0,0) mean 0
% SubSSC6: ARIMA(0,1,0)
% SubSSC7: ARIMA(1,2,2) ar1 0.5690 ma1 -1.7640, ma2 0.9003
% Here, we load the fitted results from auto.arima
BasicFitted=csvread('BasicFittedOutput.csv',1,1);
figure(2)
for i=2:8
    subplot(7,1,i-1)
    plot(BasicFitted(:,i),'r')
    hold on
    plot(SubSSC(:,i-1),'b')
end
BasicResidual=csvread('BasicResidualOutput.csv',1,1);
figure(3)
for i=2:8
    subplot(7,1,i-1)
    plot(BasicResidual(:,i),'r')
end
figure(5)
for i=2:7
    subplot(4,2,i-1)
    qqplot(BasicResidual(:,i))
end
subplot(4,2,7)
qqplot(BasicResidual(3:end,8))
%% The pacf curves are plotted for the decomposed series for our augmented regression
figure(6)
for i=1:7
    subplot(3,3,i)
    parcorr(SubSSC(:,i),'NumSTD',3)
end
%% Results of p-2, 6, 7, 7, 7, 7, and 7.
%% SubSSC4 and SubSSC5: ridge regression with augmented predictors
% SubSSC4
SubSSC4=SubSSC(:,4);
SubSSC4_Predictors=[SubSSC4(1:1883),SubSSC4(2:1884),SubSSC4(3:1885),SubSSC4(4:1886),...
    SubSSC4(5:1887),SubSSC4(6:1888),SubSSC4(7:1889)];
SubSSC4_Repsonse=SubSSC4(8:1890);
SubSSC4_Model = fitrsvm(SubSSC4_Predictors,SubSSC4_Repsonse,'KernelFunction','linear');
SubSSC4_Fitted=predict(SubSSC4_Model,SubSSC4_Predictors);
SubSSC4_Residual=SubSSC4_Repsonse-SubSSC4_Fitted;
figure(7)
subplot(2,2,1)
qqplot(SubSSC4_Residual)
subplot(2,2,2)
plot(SubSSC4_Fitted,'b')
hold on
plot(SubSSC4_Repsonse,'r')
% SubSSC5
SubSSC5=SubSSC(:,5);
SubSSC5_Predictors=[SubSSC5(1:1883),SubSSC5(2:1884),SubSSC5(3:1885),SubSSC5(4:1886),...
    SubSSC5(5:1887),SubSSC5(6:1888),SubSSC5(7:1889)];
SubSSC5_Repsonse=SubSSC5(8:1890);
SubSSC5_Model = fitrsvm(SubSSC5_Predictors,SubSSC5_Repsonse,'KernelFunction','linear');
SubSSC5_Fitted=predict(SubSSC5_Model,SubSSC5_Predictors);
SubSSC5_Residual=SubSSC5_Repsonse-SubSSC5_Fitted;
subplot(2,2,3)
qqplot(SubSSC5_Residual)
subplot(2,2,4)
plot(SubSSC5_Fitted,'b')
hold on
plot(SubSSC5_Repsonse,'r')
%% Data modelling
% Why do we use the ridge regession?
% According to the PACF plots, the multicollinearity is presented among
% each preidctors.
%% The robust ridge regession with autgmented predictors are proposed for the SubSSC1, SubSSC1-3
%%
%% The least squared ridge regression with autogmented predictors are proposed for the SubSSC1-7
%% The lncosh ridge regression is firstly proposed for each sub-series modelling, which can approximate to the l1-norm loss,
% l2-norm loss and huber loss function by a tuning parameter.
%%
% Now, the basic ridge regression is employed for each SubSSC series
% modelling in R
figure(8)
%% SubSSC1
[SubSSC1_Train_Input, SubSSC1_Train_Response,SubSSC1_Test_Input, SubSSC1_Test_Response]=Lag2_Reconstruct(SubSSC(:,1)');
SubSSC1ridge=csvread('SubSSC1ridge.csv',1,1);%predictions
SubSSC1ridgeRes=SubSSC1_Train_Response-csvread('SubSSC1res.csv',1,1);%res
for i=1:6
subplot(7,6,i)
 hist(SubSSC1ridgeRes(:,i),20,'c')
end
%% SubSSC2
[SubSSC2_Train_Input, SubSSC2_Train_Response,SubSSC2_Test_Input, SubSSC2_Test_Response]=Lag6_Reconstruct(SubSSC(:,2)');
SubSSC2ridge=csvread('SubSSC2ridge.csv',1,1);%predictions
SubSSC2ridgeRes=SubSSC2_Train_Response-csvread('SubSSC2res.csv',1,1);%res
for i=1:6
subplot(7,6,6+i)
 hist(SubSSC2ridgeRes(:,i),20,'c')
end
%% SubSSC3
[SubSSC3_Train_Input, SubSSC3_Train_Response,SubSSC3_Test_Input, SubSSC3_Test_Response]=Lag7_Reconstruct(SubSSC(:,3)');
SubSSC3ridge=csvread('SubSSC3ridge.csv',1,1);%predictions
SubSSC3ridgeRes=SubSSC3_Train_Response-csvread('SubSSC3res.csv',1,1);%res
for i=1:6
subplot(7,6,12+i)
 hist(SubSSC3ridgeRes(:,i),20,'c')
end
%% SubSSC4
[SubSSC4_Train_Input, SubSSC4_Train_Response,SubSSC4_Test_Input, SubSSC4_Test_Response]=Lag7_Reconstruct(SubSSC(:,4)');
SubSSC4ridge=csvread('SubSSC4ridge.csv',1,1);%predictions
SubSSC4ridgeRes=SubSSC4_Train_Response-csvread('SubSSC4res.csv',1,1);%res
for i=1:6
subplot(7,6,18+i)
 hist(SubSSC4ridgeRes(:,i),20,'c')
end
%% SubSSC5
[SubSSC5_Train_Input, SubSSC5_Train_Response,SubSSC5_Test_Input, SubSSC5_Test_Response]=Lag7_Reconstruct(SubSSC(:,5)');
SubSSC5ridge=csvread('SubSSC5ridge.csv',1,1);%predictions
SubSSC5ridgeRes=SubSSC5_Train_Response-csvread('SubSSC5res.csv',1,1);%res
for i=1:6
subplot(7,6,24+i)
 hist(SubSSC5ridgeRes(:,i),20,'c')
end
%% SubSSC6
[SubSSC6_Train_Input, SubSSC6_Train_Response,SubSSC6_Test_Input, SubSSC6_Test_Response]=Lag7_Reconstruct(SubSSC(:,6)');
SubSSC6ridge=csvread('SubSSC6ridge.csv',1,1);%predictions
SubSSC6ridgeRes=SubSSC6_Train_Response-csvread('SubSSC6res.csv',1,1);%res
for i=1:6
subplot(7,6,30+i)
 hist(SubSSC6ridgeRes(:,i),20,'c')
end
%% SubSSC7
[SubSSC7_Train_Input, SubSSC7_Train_Response,SubSSC7_Test_Input, SubSSC7_Test_Response]=Lag7_Reconstruct(SubSSC(:,7)');
SubSSC7ridge=csvread('SubSSC7ridge.csv',1,1);%predictions
SubSSC7ridgeRes=SubSSC7_Train_Response-csvread('SubSSC7res.csv',1,1);%res
for i=1:6
subplot(7,6,36+i)
 hist(SubSSC7ridgeRes(:,i),20,'c')
end
% qqplot
figure(9)
for i=1:6
 subplot(7,6,i)
 qqplot(SubSSC1ridgeRes(:,i))
end
for i=1:6
 subplot(7,6,6+i)
 qqplot(SubSSC2ridgeRes(:,i))
end
for i=1:6
 subplot(7,6,12+i)
 qqplot(SubSSC3ridgeRes(:,i))
end
for i=1:6
 subplot(7,6,18+i)
 qqplot(SubSSC4ridgeRes(:,i))
end
for i=1:6
 subplot(7,6,24+i)
 qqplot(SubSSC5ridgeRes(:,i))
end
for i=1:6
 subplot(7,6,30+i)
 qqplot(SubSSC6ridgeRes(:,i))
end
for i=1:6
 subplot(7,6,36+i)
 qqplot(SubSSC7ridgeRes(:,i))
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Now, we will use the robust lncosh ridge regression to mdoel Sub1-3
% SubSSC1
%Total_SubSSC1_Train_Preds=[];
%Total_SubSSC1_Test_Preds=[];
%A=[];
%for Step=1:6
%    Step
%[Optimal_Beta, Optimal_Kappa, Optimal_Lambda]=Lncosh_Ridge(SubSSC1_Train_Input,SubSSC1_Train_Response(:,Step));
%[n_train,~]=size(SubSSC1_Train_Input);
%[n_test,~]=size(SubSSC1_Test_Input);
%A=[A;Optimal_Beta];
%SubSSC1_Test_Preds=[SubSSC1_Test_Input,ones(n_test,1)]*Optimal_Beta';
%Total_SubSSC1_Test_Preds=[Total_SubSSC1_Test_Preds,SubSSC1_Test_Preds];
%SubSSC1_Train_Preds=[SubSSC1_Train_Input,ones(n_train,1)]*Optimal_Beta';
%Total_SubSSC1_Train_Preds=[Total_SubSSC1_Train_Preds,SubSSC1_Train_Preds];
%SubSSC1_Optimal_Kappa(Step)=Optimal_Kappa;
%SubSSC1_Optimal_Lambda(Step)=Optimal_Lambda;
%end
%A
%SubSSC2
%Total_SubSSC2_Train_Preds=[];
%Total_SubSSC2_Test_Preds=[];
%B=[];
%for Step=1:6
%    Step
%[Optimal_Beta, Optimal_Kappa, Optimal_Lambda]=Lncosh_Ridge(SubSSC2_Train_Input,SubSSC2_Train_Response(:,Step));
%[n_test,~]=size(SubSSC2_Test_Input);
%[n_train,~]=size(SubSSC2_Train_Input);
%B=[B;Optimal_Beta];
%SubSSC2_Test_Preds=[SubSSC2_Test_Input,ones(n_test,1)]*Optimal_Beta';
%Total_SubSSC2_Test_Preds=[Total_SubSSC2_Test_Preds,SubSSC2_Test_Preds];
%SubSSC2_Train_Preds=[SubSSC2_Train_Input,ones(n_train,1)]*Optimal_Beta';
%Total_SubSSC2_Train_Preds=[Total_SubSSC2_Train_Preds,SubSSC2_Train_Preds];
%SubSSC2_Optimal_Kappa(Step)=Optimal_Kappa;
%SubSSC2_Optimal_Lambda(Step)=Optimal_Lambda;
%end
%B
%SubSSC3
%Total_SubSSC3_Train_Preds=[];
%Total_SubSSC3_Test_Preds=[];
%C=[];
%for Step=1:6
%    Step
%[Optimal_Beta, Optimal_Kappa, Optimal_Lambda]=Lncosh_Ridge(SubSSC3_Train_Input,SubSSC3_Train_Response(:,Step));
%[n_test,~]=size(SubSSC3_Test_Input);
%[n_train,~]=size(SubSSC3_Train_Input);
%C=[C;Optimal_Beta];
%SubSSC3_Test_Preds=[SubSSC3_Test_Input,ones(n_test,1)]*Optimal_Beta';
%Total_SubSSC3_Test_Preds=[Total_SubSSC3_Test_Preds,SubSSC3_Test_Preds];
%SubSSC3_Train_Preds=[SubSSC3_Train_Input,ones(n_train,1)]*Optimal_Beta';
%Total_SubSSC3_Train_Preds=[Total_SubSSC3_Train_Preds,SubSSC3_Train_Preds];
%SubSSC3_Optimal_Kappa(Step)=Optimal_Kappa;
%SubSSC3_Optimal_Lambda(Step)=Optimal_Lambda;
%end
%C
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load Total_Robust_Preds
%%%%%%%%%%%%%%%%%%%%%
%Robust_SubSSC1_Preds
figure(10)
for i=1:6
    subplot(6,1,i)
    plot(Total_SubSSC1_Test_Preds(:,i),'b')
    hold on
    plot(SubSSC1_Test_Response(:,i),'k')
end
figure(11)
for i=1:6
    subplot(3,2,i)
    qqplot(SubSSC1_Test_Response(:,i)-Total_SubSSC1_Test_Preds(:,i))
end
%Robust_SubSSC2_Preds
figure(12)
for i=1:6
    subplot(6,1,i)
    plot(Total_SubSSC2_Test_Preds(:,i),'b')
    hold on
    plot(SubSSC2_Test_Response(:,i),'k')
end
figure(13)
for i=1:6
    subplot(3,2,i)
    qqplot(SubSSC2_Test_Response(:,i)-Total_SubSSC2_Test_Preds(:,i))
end
%Robust_SubSSC3_Preds
figure(14)
for i=1:6
    subplot(6,1,i)
    plot(Total_SubSSC3_Test_Preds(:,i),'b')
    hold on
    plot(SubSSC3_Test_Response(:,i),'k')
end
figure(15)
for i=1:6
    subplot(3,2,i)
    qqplot(SubSSC3_Test_Response(:,i)-Total_SubSSC3_Test_Preds(:,i))
end
%% Predictions
RidgePreds=SubSSC1ridge+SubSSC2ridge+SubSSC3ridge+SubSSC4ridge+SubSSC5ridge+...
    SubSSC6ridge+SubSSC7ridge;
RobustRidgePreds=Total_SubSSC1_Test_Preds+Total_SubSSC2_Test_Preds+Total_SubSSC3_Test_Preds+...
    SubSSC4ridge+SubSSC5ridge+SubSSC6ridge+SubSSC7ridge;
TestResponses=SubSSC1_Test_Response+SubSSC2_Test_Response+SubSSC3_Test_Response+SubSSC4_Test_Response+...
    SubSSC5_Test_Response+SubSSC6_Test_Response+SubSSC7_Test_Response;
figure(16)
for i=1:6
    subplot(6,1,i)
    plot(TestResponses(:,i),'k')
    hold on
    plot(RidgePreds(:,i),'b')
    hold on
    plot(RobustRidgePreds(:,i),'r')
end
%% Error Index
[mean((TestResponses-RobustRidgePreds).^2)',mean(abs(TestResponses-RobustRidgePreds))',mean(abs(TestResponses-RobustRidgePreds)./TestResponses)']
[mean((TestResponses-RidgePreds).^2)',mean(abs(TestResponses-RidgePreds))',mean(abs(TestResponses-RidgePreds)./TestResponses)']    

%% 
RidgeSSC1Fitted=csvread('SubSSC1res.csv',1,1);
RidgeSSC2Fitted=csvread('SubSSC2res.csv',1,1);
RidgeSSC3Fitted=csvread('SubSSC3res.csv',1,1);
load All_Fitted_Preds
figure(17)
for i=1:6
subplot(3,6,i)
histogram(SubSSC1_Train_Response(:,i)-Total_SubSSC1_Train_Preds(:,i),'BinWidth',0.03,'FaceColor','k')
hold on
histogram(SubSSC1_Train_Response(:,i)-RidgeSSC1Fitted(:,i),'BinWidth',0.03,'FaceColor','w')
end
for i=1:6
subplot(3,6,6+i)
histogram(SubSSC2_Train_Response(:,i)-Total_SubSSC2_Train_Preds(:,i),'BinWidth',0.02,'FaceColor','k')
hold on
histogram(SubSSC2_Train_Response(:,i)-RidgeSSC2Fitted(:,i),'BinWidth',0.02,'FaceColor','w')
end
for i=1:6
subplot(3,6,12+i)
histogram(SubSSC3_Train_Response(:,i)-Total_SubSSC3_Train_Preds(:,i),'BinWidth',0.01,'FaceColor','k')
hold on
histogram(SubSSC3_Train_Response(:,i)-RidgeSSC3Fitted(:,i),'BinWidth',0.01,'FaceColor','w')
end
%%test
figure(18)
for i=1:6
subplot(6,3,3*(i-1)+1)
plot(SubSSC1_Test_Response(:,i),'k')
hold on
plot(Total_SubSSC1_Test_Preds(:,i),'r')
hold on
plot(SubSSC1ridge(:,i),'b')
end
for i=1:6
subplot(6,3,3*(i-1)+2)
plot(SubSSC2_Test_Response(:,i),'k')
hold on
plot(Total_SubSSC2_Test_Preds(:,i),'r')
hold on
plot(SubSSC2ridge(:,i),'b')
end
for i=1:6
subplot(6,3,3*(i-1)+3)
plot(SubSSC3_Test_Response(:,i),'k')
hold on
plot(Total_SubSSC3_Test_Preds(:,i),'r')
hold on
plot(SubSSC3ridge(:,i),'b')
end
%%
parcorr(SSC_Average,'NumLags',20,'NumSTD',3)
%%
SingData=[SSC(1:1867)',SSC(3:1869)',SSC(5:1871)',SSC(6:1872)',SSC(10:1876)',SSC(11:1877)'...
    SSC(13:1879)',SSC(14:1880)',SSC(18:1884)',SSC(19:1885)',SSC(20:1886)',SSC(21:1887)',...
    SSC(22:1888)',SSC(23:1889)',SSC(24:1890)'];
SingDataTrainInput=SingData(1:1657,1:9);
SingDataTrainResponse=SingData(1:1657,10:end);
SingDataTestInput=SingData(1658:end,1:9);
SingDataTestResponse=SingData(1658:end,10:end);
%% single least squared linear regression
for step=1:6
ls_model = fitlm(SingDataTrainInput,SingDataTrainResponse(:,step),'interactions');
LSPreds=predict(ls_model,SingDataTestInput);
MSE(step)=mean((SingDataTestResponse(:,step)-LSPreds).^2);
MAE(step)=mean(abs(SingDataTestResponse(:,step)-LSPreds));
MRE(step)=mean(abs(SingDataTestResponse(:,step)-LSPreds)./SingDataTestResponse(:,step));
end
SingleLeastedLR_Results=[MSE',MAE',MRE']
%% single lncosh ridge regression
for step=1:6
[Optimal_Beta, Optimal_Kappa, Optimal_Lambda]=Lncosh_Ridge(SingDataTrainInput,SingDataTrainResponse(:,step));
[n_test,~]=size(SingDataTestInput);
SingleLncoshRidge_Preds=[SingDataTestInput,ones(n_test,1)]*Optimal_Beta';
MSE(step)=mean((SingDataTestResponse(:,step)-SingleLncoshRidge_Preds).^2);
MAE(step)=mean(abs(SingDataTestResponse(:,step)-SingleLncoshRidge_Preds));
MRE(step)=mean(abs(SingDataTestResponse(:,step)-SingleLncoshRidge_Preds)./SingDataTestResponse(:,step));
end
%% emd-ridge
for step=1:6
EMD_Ridge1=fitlm(SubSSC1_Train_Input,SubSSC1_Train_Response(:,step),'interactions');
EMD_Ridge2=fitlm(SubSSC2_Train_Input,SubSSC2_Train_Response(:,step),'interactions');
EMD_Ridge3=fitlm(SubSSC3_Train_Input,SubSSC3_Train_Response(:,step),'interactions');
EMD_Ridge4=fitlm(SubSSC4_Train_Input,SubSSC4_Train_Response(:,step),'interactions');
EMD_Ridge5=fitlm(SubSSC5_Train_Input,SubSSC5_Train_Response(:,step),'interactions');
EMD_Ridge6=fitlm(SubSSC6_Train_Input,SubSSC6_Train_Response(:,step),'interactions');
EMD_Preds1=predict(EMD_Ridge1,SubSSC1_Test_Input);
EMD_Preds2=predict(EMD_Ridge2,SubSSC2_Test_Input);
EMD_Preds3=predict(EMD_Ridge3,SubSSC3_Test_Input);
EMD_Preds4=predict(EMD_Ridge4,SubSSC4_Test_Input);
EMD_Preds5=predict(EMD_Ridge5,SubSSC5_Test_Input);
EMD_Preds6=predict(EMD_Ridge6,SubSSC6_Test_Input);
EMD_Preds7=predict(EMD_Ridge7,SubSSC7_Test_Input);
EMD_Preds=EMD_Preds1+EMD_Preds2+EMD_Preds3+EMD_Preds4+EMD_Preds5++SubSSC6_Test_Response+SubSSC7_Test_Response;
MSE(step)=mean((SingDataTestResponse(:,step)-EMD_Preds).^2);
MAE(step)=mean(abs(SingDataTestResponse(:,step)-EMD_Preds));
MRE(step)=mean(abs(SingDataTestResponse(:,step)-EMD_Preds)./SingDataTestResponse(:,step));
end

