clear all
close all
addpath('../PolBSP and PolSCENE');
%rng('default'); % For reproducibility

%Load data

videoPrefix='overhead';
%videoPrefix='motorway';
dataFile=sprintf('../Tracking/%s_tracking.mat',videoPrefix);
snapshotFile=sprintf('../../Vídeos/%s_snapshot.png',videoPrefix);
barriersFile=sprintf('%s_barriers.mat',videoPrefix);
load(dataFile); %variable CollectedData contains positions and speeds
load(barriersFile); %variable BarriersData contains positions of barriers
%to be added (cell array)(barriers are handmade so far)

%Show snapshot of the video 

snapshot = imread(snapshotFile);
h=imshow(snapshot);hold on;

%break

%Create barriers from the barriers file 

S=PolSCENEInitialize();
numBarriers=size(BarriersData,2);
for i=1:numBarriers
    S=PolSCENEAddBarrier(S,BarriersData{i});
end
S=PolSCENEComputeVisibilityGraph(S);
PolBSPRender(S.PolBSPTree);
%PolSCENEShow(S);

%Create samples from the tracking file
Samples=CollectedData(1:2,:);
NumSamples=size(Samples,2);
scatter(Samples(1,:),Samples(2,:));

%Parameters

RowsCols=6;
Parameters.NumRowsMap=RowsCols;
Parameters.NumColsMap=RowsCols;
Parameters.NumSteps=500;      
Parameters.Topology='Square';
Parameters.Toroidal=0; 
Parameters.InitialLearningRate=0.2;
Parameters.MaxRadius=RowsCols/4;
Parameters.ConvergenceLearningRate=0.05;
Parameters.ConvergenceRadius=0.1;

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

