function [TrainInput,TrainRespone,TestInput,TestRespone]=Lag6_Reconstruct(data)
A=[];
for i=1:1879
    a=data(i:i+11);
    A=[A;a];
end
inputs=A(:,1:6);
response=A(:,7:12);
%
TrainInput=inputs(1:1669,:);
TrainRespone=response(1:1669,:);
TestInput=inputs(1670:end,:);
TestRespone=response(1670:end,:);
end