clear all
close all
addpath('../PolBSP and PolSCENE');
%rng('default'); % For reproducibility
%Create barriers

S=PolSCENEInitialize();
d=0.5;
S=PolSCENEAddBarrier(S,[-1-d,-1-d;-1-d,1+d;1+d,1+d;1+d,-1-d]);
S=PolSCENEComputeVisibilityGraph(S);
h=figure;hold on;
PolBSPRender(S.PolBSPTree);
%PolSCENEShow(S);

%Parameters

RowsCols=4;
Parameters.NumRowsMap=1;
Parameters.NumColsMap=RowsCols^2;
Parameters.NumSteps=200;      
Parameters.Topology='Square';
Parameters.Toroidal=0; 
Parameters.InitialLearningRate=0.2;
Parameters.MaxRadius=RowsCols/4;
Parameters.ConvergenceLearningRate=0.05;
Parameters.ConvergenceRadius=0.1;

%Create samples
NumSamples=1000;
SamplesX=[unifrnd(-2,-1-d,1,500),unifrnd(-2,1+d,1,500),unifrnd(1+d,2,1,500),unifrnd(-1-d,2,1,500)];
SamplesY=[unifrnd(-2,1+d,1,500),unifrnd(1+d,2,1,500),unifrnd(-1-d,2,1,500),unifrnd(-2,-1-d,1,500)];
Samples=[SamplesX;SamplesY];
scatter(SamplesX,SamplesY);

%Standard SOFM

Model=TrainSOFM(Samples,Parameters);

%Forbidden Region SOM (FRSOM)

ModelFR=TrainFRSOFM(Samples,Parameters,S);

%Comparison
[Winners,Errors,TopologyError,OffendingSamples]=CompetitionSOFMMEX(Model,Samples);
[WinnersFR,ErrorsFR,TopologyErrorFR,OffendingSamplesFR]=CompetitionSOFMMEX(ModelFR,Samples);
MSE=sum(Errors)/NumSamples
MSEFR=sum(ErrorsFR)/NumSamples
scatter(reshape(Model.Prototypes(1,:,:),[1,RowsCols^2]),reshape(Model.Prototypes(2,:,:),[1,RowsCols^2]),'g');
scatter(reshape(ModelFR.Prototypes(1,:,:),[1,RowsCols^2]),reshape(ModelFR.Prototypes(2,:,:),[1,RowsCols^2]),'r');

