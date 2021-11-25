function [TrainInput,TrainRespone,TestInput,TestRespone]=Lag7_Reconstruct(data)
A=[];
for i=1:1878
    a=data(i:i+12);
    A=[A;a];
end
inputs=A(:,1:7);
response=A(:,8:13);
%
TrainInput=inputs(1:1668,:);
TrainRespone=response(1:1668,:);
TestInput=inputs(1669:end,:);
TestRespone=response(1669:end,:);
end