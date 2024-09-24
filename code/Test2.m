clear all
close all
addpath('../PolBSP and PolSCENE');
rng('default'); % For reproducibility
%Create barriers

S=PolSCENEInitialize();
S=PolSCENEAddBarrier(S,[-3,-3;-3,1;-1,1;-1,-3]);
S=PolSCENEAddBarrier(S,[-3,1;-3,3;1,3;1,1]);
S=PolSCENEAddBarrier(S,[1,3;3,3;3,-1;1,-1]);
S=PolSCENEAddBarrier(S,[-0.5,-1;3,-1;3,-2.5;-0.5,-2.5]);
S=PolSCENEComputeVisibilityGraph(S);
h=figure;hold on;
PolBSPRender(S.PolBSPTree); 
%PolSCENEShow(S);

%Parameters

RowsCols=4;
Parameters.NumRowsMap=RowsCols;
Parameters.NumColsMap=RowsCols;
Parameters.NumSteps=200;      
Parameters.Topology='Square';
Parameters.Toroidal=0; 
Parameters.InitialLearningRate=0.2;
Parameters.MaxRadius=RowsCols/4;
Parameters.ConvergenceLearningRate=0.05;
Parameters.ConvergenceRadius=0.1;

%Create samples
NumSamples=1000;
SamplesX=[unifrnd(-0.5,0.5,1,500),4.+unifrnd(-0.5,0.5,1,500),unifrnd(-0.5,4.5,1,500)];
SamplesY=[unifrnd(-0.5,0.5,1,1000),unifrnd(-3,-2.5,1,500)];
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

