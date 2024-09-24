clear all
warning off

NumFolds=10;
NumSteps=100000;

ListMethods={'Square','Hex','Tri','Cairo','Prismatic'};
ListMapSizes={'Four','Eight','Sixteen','ThirtyTwo'};
ListDatasets={'Baboon','F16','House','Lake','Lena','Peppers'};
MapSizes=[4 8 16 32];
NumMethods=numel(ListMethods);
OptimOptions = optimset('fminsearch');
OptimOptions = optimset(OptimOptions,'Display','none','MaxFunEvals',100,'TolX',0.001,'TolFun',0.001);

MyArrayID=getenv('SLURM_ARRAYID');
if ~isempty(MyArrayID)
     ArrayIDNumber=sscanf(MyArrayID,'%d');
     fprintf('\r\nTask %d starting.\r\n',ArrayIDNumber);
else
     ArrayIDNumber=0;
end

pause(5*rand(1))
if ~exist('./ResultsImagesTop.mat','file')
    Results=[];
    save('./ResultsImagesTop.mat','Results');
else
    load('./ResultsImagesTop.mat','Results');    
end

KeepOn=1;
while KeepOn
    load('./ResultsImagesTop.mat','Results');
    
    % Look for undone jobs
    JobDone=zeros(size(ListMapSizes,2),size(ListDatasets,2));
    for NdxMapSize=1:size(ListMapSizes,2)
        MapSizeName=ListMapSizes{NdxMapSize};
        if isfield(Results,MapSizeName)
            for NdxDataset=1:size(ListDatasets,2)
                DatasetName=ListDatasets{NdxDataset};
                if isfield(Results.(MapSizeName),DatasetName)
                    JobDone(NdxMapSize,NdxDataset)=1;
                end
            end
        end
    end
    
    % Randomly select an undone job to do it
    Undone=find(JobDone==0);
    if isempty(Undone)
        KeepOn=0;
        continue;
    end
    rand('twister',sum(100*clock)+ArrayIDNumber);
    MyJob=Undone(ceil(numel(Undone)*rand(1)));
    rand('twister',5489); % This is the default random seed
    
    [NdxMapSize NdxDataset]=ind2sub([size(ListMapSizes,2) size(ListDatasets,2)],MyJob);
    MapSizeName=ListMapSizes{NdxMapSize};
    DatasetName=ListDatasets{NdxDataset};
    MyMapSize=MapSizes(NdxMapSize);

    fprintf('\r\nProcessing dataset %s with map size %d.\r\n',DatasetName,MyMapSize);
    MyResult=[];
    
    % Get dataset
    FileName=sprintf('./OriginalImages/%s.png',DatasetName);
    OriginalImage=double(imread(FileName))/255;
    Samples=reshape(OriginalImage,[size(OriginalImage,1)*size(OriginalImage,2) 3])';    
    NumSamples=size(Samples,2);
    RandIndicesOptim=ceil(NumFolds*rand(1,NumSamples));
    TrainSamplesOptim=Samples(:,RandIndicesOptim~=1);
    ValidationSamplesOptim=Samples(:,RandIndicesOptim==1);
    GlobalMean=mean(TrainSamplesOptim,2);
    MSE_Zero=mean(sum((TrainSamplesOptim-repmat(GlobalMean,1,size(TrainSamplesOptim,2))).^2,1));
    RandIndicesCV=ceil(NumFolds*rand(1,NumSamples));
    
    % Run the tests
    for NdxMethod=1:NumMethods
        MyMethod=ListMethods{NdxMethod};
        
        % Optimize the method parameters
        Parameters.NumRowsMap=MyMapSize;
        Parameters.NumColsMap=MyMapSize;
        Parameters.NumSteps=NumSteps;      
        Parameters.Topology=MyMethod;
        Parameters.Toroidal=0;        
        LowerBounds=[0.01 MyMapSize/16 1.0e-4 1.0e-3];
        UpperBounds=[0.99 2*MyMapSize 0.2 2];
        InitialSolution=[0.4 MyMapSize/2 0.01 1];
        [FoundSolution,FoundValue]=fminsearch(@(x) ObjectiveFunction(x,MyMethod,Parameters,TrainSamplesOptim,ValidationSamplesOptim,MSE_Zero),...
            InitialSolution,OptimOptions);
        InitialValue=ObjectiveFunction(InitialSolution,MyMethod,Parameters,TrainSamplesOptim,ValidationSamplesOptim,MSE_Zero);
        if InitialValue<FoundValue
            % The initial solution is better
            BestSolution=InitialSolution;
        else
            % The solution provided by fminsearch is better
            BestSolution=FoundSolution;
        end
        
        % Set optimal parameters
        Parameters.InitialLearningRate=BestSolution(1);
        Parameters.MaxRadius=BestSolution(2);
        Parameters.ConvergenceLearningRate=BestSolution(3);
        Parameters.ConvergenceRadius=BestSolution(4);
    
        % Run the cross validated tests
        MyResult.(MyMethod).CPUtime.Values=zeros(NumFolds,1);
        MyResult.(MyMethod).MSE.Values=zeros(NumFolds,1);
        MyResult.(MyMethod).TE.Values=zeros(NumFolds,1);
        MyResult.(MyMethod).MSV.Values=zeros(NumFolds,1);
        MyResult.(MyMethod).MTR.Values=zeros(NumFolds,1);
        for NdxRun=1:NumFolds
            TrainSamplesCV=Samples(:,RandIndicesCV~=NdxRun);
            TestSamplesCV=Samples(:,RandIndicesCV==NdxRun);
            t=clock;
            Model=TrainSOFM(TrainSamplesCV,Parameters);
            MyResult.(MyMethod).CPUtime.Values(NdxRun)=etime(clock,t);
            [Winners,Errors,TopologyError,OffendingSamples,SampleTiedRanks]=CompetitionSOFMMEX(Model,TestSamplesCV);
            SilhouetteValues=silhouette(TestSamplesCV',Winners);
            MyResult.(MyMethod).MSE.Values(NdxRun)=mean(Errors);
            MyResult.(MyMethod).TE.Values(NdxRun)=TopologyError;  
            MyResult.(MyMethod).MSV.Values(NdxRun)=mean(SilhouetteValues);
            MyResult.(MyMethod).MTR.Values(NdxRun)=mean(SampleTiedRanks);
        end
        
        % Compute statistics
        MyResult.(MyMethod).CPUtime.Mean=mean(MyResult.(MyMethod).CPUtime.Values);
        MyResult.(MyMethod).CPUtime.StdDev=std(MyResult.(MyMethod).CPUtime.Values);
        MyResult.(MyMethod).MSE.Mean=mean(MyResult.(MyMethod).MSE.Values);
        MyResult.(MyMethod).MSE.StdDev=std(MyResult.(MyMethod).MSE.Values);
        MyResult.(MyMethod).TE.Mean=mean(MyResult.(MyMethod).TE.Values);
        MyResult.(MyMethod).TE.StdDev=std(MyResult.(MyMethod).TE.Values);
        MyResult.(MyMethod).MSV.Mean=mean(MyResult.(MyMethod).MSV.Values);
        MyResult.(MyMethod).MSV.StdDev=std(MyResult.(MyMethod).MSV.Values);
        MyResult.(MyMethod).MTR.Mean=mean(MyResult.(MyMethod).MTR.Values);
        MyResult.(MyMethod).MTR.StdDev=std(MyResult.(MyMethod).MTR.Values);        
  
    end
    
   
    % Save results
    pause(5*rand(1))
    load('./ResultsImagesTop.mat','Results');
    Results.(MapSizeName).(DatasetName)=MyResult;
    save('./ResultsImagesTop.mat','Results');    
end

% Generate report on text file
FieldNames={'CPUtime','MSE','TE','MTR'};
MyFile=fopen('./ResultsImagesTop.txt','w');

for NdxMethod=1:NumMethods
    MyMethod=ListMethods{NdxMethod};
    fprintf(MyFile,'%s\r\n\r\n',MyMethod);
    for NdxField=1:length(FieldNames)
        FieldName=FieldNames{NdxField};
        fprintf(MyFile,'%s\r\n',FieldName);
        for NdxMapSize=1:size(ListMapSizes,2)
            MapSizeName=ListMapSizes{NdxMapSize};
            if isfield(Results,MapSizeName)
                fprintf(MyFile,'%s\t',MapSizeName);
                for NdxDataset=1:size(ListDatasets,2)
                    DatasetName=ListDatasets{NdxDataset};
                    if isfield(Results.(MapSizeName),DatasetName)
                        MySubRecord=Results.(MapSizeName).(DatasetName).(MyMethod).(FieldName);
                        fprintf(MyFile,'%6.4f (%6.4f)\t',MySubRecord.Mean,MySubRecord.StdDev);
                    else
                        fprintf(MyFile,'N/D\t');                    
                    end
                end
                fprintf(MyFile,'\r\n');
            end
        end
    end
end

fclose(MyFile);









