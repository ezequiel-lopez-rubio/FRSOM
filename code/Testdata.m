clear all
close all
addpath('../PolBSP and PolSCENE');
%rng('default'); % For reproducibility

%Load data

%dataFile='From IOBIS.ORG (Delphinus delphis in Celtic Seas+North Seas).mat';
dataFile='From IOBIS.ORG (Balaenoptera physalus near Greenland and Iceland).mat';

%barriersFile='Celtic Seas+North Seas_barriers.mat';
barriersFile='Greenland+Iceland_barriers.mat';

load(dataFile); %variable LonLat contains positions (longitude and latitude) of observations
load(barriersFile); %variable BarriersData contains positions of barriers
%to be added (cell array)(barriers are handmade so far)

%Create barriers from the barriers file 

S=PolSCENEInitialize();
numBarriers=size(BarriersData,2);
for i=1:numBarriers
    S=PolSCENEAddBarrier(S,BarriersData{i});
end
S=PolSCENEComputeVisibilityGraph(S);
figure;hold;
PolBSPRender(S.PolBSPTree);
%PolSCENEShow(S);

%Create samples from the tracking file
%Samples=LonLat(:,1:2);
Samples=unique(LonLat(:,1:2),'rows');
NumSamples=size(Samples,1);
scatter(Samples(:,1),Samples(:,2));

%Parameters


RowsCols=6;
Parameters.NumRowsMap=RowsCols;
Parameters.NumColsMap=RowsCols;
Parameters.NumSteps=500;      
Parameters.Topology='Square';
Parameters.Toroidal=0; 
Parameters.InitialLearningRate=0.2;
%Parameters.MaxRadius=RowsCols/4;
Parameters.MaxRadius=RowsCols/8;
Parameters.ConvergenceLearningRate=0.05;
Parameters.ConvergenceRadius=0.1;

%BATCH WORK
NumIters=10;
outputFile='results.txt';
fid=fopen(outputFile,'w');

for i=1:NumIters
    %Standard SOFM
    Model=TrainSOFM(Samples',Parameters);

    %Forbidden Region SOM (FRSOM)
    ModelFR=TrainFRSOFM(Samples',Parameters,S);

    %Comparison
    [Winners,Errors,TopologyError,OffendingSamples]=CompetitionSOFMMEX(Model,Samples');
    [WinnersFR,ErrorsFR,TopologyErrorFR,OffendingSamplesFR]=CompetitionSOFMMEX(ModelFR,Samples');
    MSE=sum(Errors)/NumSamples
    MSEFR=sum(ErrorsFR)/NumSamples
    scatter(reshape(Model.Prototypes(1,:,:),[1,RowsCols^2]),reshape(Model.Prototypes(2,:,:),[1,RowsCols^2]),'g');
    scatter(reshape(ModelFR.Prototypes(1,:,:),[1,RowsCols^2]),reshape(ModelFR.Prototypes(2,:,:),[1,RowsCols^2]),'r');
    
    %BATCH WORK
    fprintf(fid,'%s,%s,%d,%d,%d,%s,%d,%f,%f,%f,%f,%f,%f\r\n',dataFile,barriersFile,Parameters.NumRowsMap,...
    Parameters.NumColsMap,Parameters.NumSteps,Parameters.Topology,Parameters.Toroidal,...
    Parameters.InitialLearningRate,Parameters.MaxRadius,Parameters.ConvergenceLearningRate,...
    Parameters.ConvergenceRadius,MSE,MSEFR);
end

%BATCH WORK
fclose(fid);


