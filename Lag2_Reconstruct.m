function [TrainInput,TrainRespone,TestInput,TestRespone]=Lag2_Reconstruct(data)
A=[];
for i=1:1883
    a=data(i:i+7);
    A=[A;a];
end
inputs=A(:,1:2);
response=A(:,3:8);
%
TrainInput=inputs(1:1673,:);
TrainRespone=response(1:1673,:);
TestInput=inputs(1674:end,:);
TestRespone=response(1674:end,:);
end