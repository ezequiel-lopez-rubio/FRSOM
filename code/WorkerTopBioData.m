clear all
close all
addpath('../PolBSP and PolSCENE');
%rng('default'); % For reproducibility
graphic_verbose=1; %whether show graphic data 
text_verbose=1; %whether show text data
map_verbose=1; %whether display worlwide map as background
optim_method=-1; %parameters optimization method
                %<0: no optimization, use initial solution provided
                %0: only fminsearch on all variables
                %1: first fminbnd on each variable (order can be chosen
                %below) and then fminsearch an all variables
                %2: first evaluation on grid (size can be chosen below) 
                %and then fminsearch an all variables
cross_validation=0; %whether to use (NumFolds)-folds cross validation 
                    %for competition SOM-FRSOM
                
NumSteps=1000;
RowsCols=6;
NumFolds=10;

outputFiles={'Doplhins_results.txt','Whales_results.txt','Shark_results.txt'};
dataFiles={'From IOBIS.ORG (Delphinus delphis in Celtic Seas+North Seas).mat',...
            'From IOBIS.ORG (Balaenoptera physalus near Greenland and Iceland).mat',...
            'From IOBIS.ORG (Galeocerdo cuvier near Australia).mat'};
barriersFiles={'Celtic Seas+North Seas_barriers.mat','Greenland+Iceland_barriers.mat',...
               'Australia_barriers.mat'};
tasks_order=[1,2,3];

%outputFile='results.txt';

for task_number=1:3
    
    task_current=tasks_order(task_number);
    outputFile=outputFiles{task_current};
    dataFile=dataFiles{task_current};
    barriersFile=barriersFiles{task_current};
      
    %Load data           
    load(dataFile); %variable LonLat contains positions (longitude and latitude) of observations
    load(barriersFile); %variable BarriersData contains positions of barriers

    %Add map to the background
    if graphic_verbose
        h=figure;hold;
        if map_verbose
            %seamap=imread('..\..\Data\SeaElevation(NOAA).jpg');
            %seamap=imread('..\..\Data\color_etopo1_ice_low(NOAA).jpg');               
            seamap=imread('color_etopo1_ice_low(NOAA).jpg');               
            imagesc([-180,180],[90,-90],seamap);
        end
    end

    %Create barriers from the barriers file 
    S=PolSCENEInitialize();
    numBarriers=size(BarriersData,2);
    for i=1:numBarriers
        S=PolSCENEAddBarrier(S,BarriersData{i});
    end
    S=PolSCENEComputeVisibilityGraph(S);
    if graphic_verbose 
        PolBSPRender(S.PolBSPTree);
    end
    if text_verbose
        PolSCENEShow(S);
    end

    %Create samples
    Samples=unique(LonLat,'rows')';
    %Samples=LonLat';
    NumSamples=size(Samples,2);
    if graphic_verbose    
        scatter(Samples(1,:),Samples(2,:));
    end

    fprintf('DataFile: "%s"\nBarriersFile: "%s"\nOutputFile: "%s"\n',dataFile,barriersFile,outputFile);    
    fprintf('Optimization method: %d\nCross validation: %d\n',optim_method,cross_validation);
    
    %Parameters
    Parameters.NumRowsMap=RowsCols;
    Parameters.NumColsMap=RowsCols;
    Parameters.NumSteps=NumSteps;      
    Parameters.Topology='Square';
    Parameters.Toroidal=0; 

    % Optimize the method parameters 
    fprintf('Optimizing parameters...\n');
    RandIndicesOptim=ceil((NumFolds+1)*rand(1,NumSamples));
    TrainSamplesOptim=Samples(:,RandIndicesOptim~=1);
    ValidationSamplesOptim=Samples(:,RandIndicesOptim==1);
    GlobalMean=mean(TrainSamplesOptim,2);
    MSE_Zero=mean(sum((TrainSamplesOptim-repmat(GlobalMean,1,size(TrainSamplesOptim,2))).^2,1));
    OptimOptions = optimset('fminsearch');
    OptimOptions = optimset(OptimOptions,'Display','none','MaxFunEvals',100,'MaxIter',100,'TolX',0.0001,'TolFun',0.0001);
    LowerBounds=[0.01 RowsCols/10 0.01 0.05];
    UpperBounds=[1 RowsCols 1 RowsCols];
    InitialSolution=[0.2 0.75 0.05 0.1];
    InitialValue=ObjectiveFunction(InitialSolution,Parameters,TrainSamplesOptim,ValidationSamplesOptim,MSE_Zero);
    if text_verbose
        fprintf('Current solution: %f,%f,%f,%f - Current Value: %f\n',InitialSolution(1),InitialSolution(2),...
            InitialSolution(3),InitialSolution(4),InitialValue);
    end  
    %InitialSolution=[0.2 RowsCols/8 0.05 0.1];
    %LowerBounds=[0.01 RowsCols/16 1.0e-4 1.0e-3];
    %UpperBounds=[0.5 RowsCols 0.5 2];

    if optim_method==1
        % First, we optimize on each variable alone fixing the rest (depends on
        % the order of course)
        optim_method1_order=[4,2,1,3];%choose here the order
        BestSolution=InitialSolution;
        for i=1:4
            j=optim_method1_order(i);
            switch j
                case 1
                    % Minimum on InitialLearningRate
                    [FoundSolution,FoundValue]=fminbnd(@(x) ObjectiveFunction([x,BestSolution(2),BestSolution(3),...
                        BestSolution(4)],Parameters,TrainSamplesOptim,ValidationSamplesOptim,MSE_Zero),0,1,OptimOptions);
                    BestSolution(1)=FoundSolution
                case 2
                    % Minimum on MaxRadius
                    [FoundSolution,FoundValue]=fminbnd(@(x) ObjectiveFunction([BestSolution(1),x,BestSolution(3),...
                        BestSolution(4)],Parameters,TrainSamplesOptim,ValidationSamplesOptim,MSE_Zero),0,RowsCols,OptimOptions);
                    BestSolution(2)=FoundSolution
                case 3
                    %Minimum in ConvergenceLearningRate
                    [FoundSolution,FoundValue]=fminbnd(@(x) ObjectiveFunction([BestSolution(1),BestSolution(2),x,...
                        BestSolution(4)],Parameters,TrainSamplesOptim,ValidationSamplesOptim,MSE_Zero),0,1,OptimOptions);
                    BestSolution(3)=FoundSolution
                case 4
                    %Minimum in ConvergenceRadius
                    [FoundSolution,FoundValue]=fminbnd(@(x) ObjectiveFunction([BestSolution(1),BestSolution(2),...
                        BestSolution(3),x],Parameters,TrainSamplesOptim,ValidationSamplesOptim,MSE_Zero),0,RowsCols,OptimOptions);        
                    BestSolution(4)=FoundSolution
            end
        end
        InitialSolution=BestSolution;
        InitialValue=FoundValue;
        if text_verbose
            fprintf('Current solution: %f,%f,%f,%f - Current Value: %f\n',InitialSolution(1),InitialSolution(2),...
                InitialSolution(3),InitialSolution(4),InitialValue);
        end      
    elseif optim_method==2
        % First, we optimize by evaluating on grid
        BestSolution=InitialSolution;
        BestValue=InitialValue;
        GridPoints=5;%Watch out! (GridPoints+1)^4-growth on number of evaluations below
        [Grid1,Grid2,Grid3,Grid4]=ndgrid([LowerBounds(1):(UpperBounds(1)-LowerBounds(1))/GridPoints:UpperBounds(1)],...
            [LowerBounds(2):(UpperBounds(2)-LowerBounds(2))/GridPoints:UpperBounds(2)],...
            [LowerBounds(3):(UpperBounds(3)-LowerBounds(3))/GridPoints:UpperBounds(3)],...
            [LowerBounds(4):(UpperBounds(4)-LowerBounds(4))/GridPoints:UpperBounds(4)]);
        GridSize=size(Grid1,1)*size(Grid1,2)*size(Grid1,3)*size(Grid1,4);
        GridLine1=reshape(Grid1,GridSize,[]);GridLine2=reshape(Grid2,GridSize,[]);
        GridLine3=reshape(Grid3,GridSize,[]);GridLine4=reshape(Grid4,GridSize,[]);
        for i=1:GridSize
            FoundSolution=[GridLine1(i),GridLine2(i),GridLine3(i),GridLine4(i)];
            FoundValue=ObjectiveFunction(FoundSolution,Parameters,TrainSamplesOptim,ValidationSamplesOptim,MSE_Zero);
            if FoundValue<BestValue
                BestValue=FoundValue;
                BestSolution=FoundSolution;
            end
        end
        InitialSolution=BestSolution;
        InitialValue=BestValue;
        if text_verbose
            fprintf('Current solution: %f,%f,%f,%f - Current Value: %f\n',InitialSolution(1),InitialSolution(2),...
                InitialSolution(3),InitialSolution(4),InitialValue);
        end 
    end    

    if optim_method>=0
        % Second, we optimize near the proposed minimum with fminsearch
        [BestSolution,BestValue]=fminsearch(@(x) ObjectiveFunction(x,Parameters,TrainSamplesOptim,ValidationSamplesOptim,MSE_Zero),...
            InitialSolution,OptimOptions);
        if text_verbose
            fprintf('Current solution: %f,%f,%f,%f - Current Value: %f\n',BestSolution(1),BestSolution(2),...
                BestSolution(3),BestSolution(4),BestValue);
        end 
    else %no optimization
        BestSolution=InitialSolution;
    end

    % Set optimal parameters
    Parameters.InitialLearningRate=BestSolution(1);
    Parameters.MaxRadius=BestSolution(2);
    Parameters.ConvergenceLearningRate=BestSolution(3);
    Parameters.ConvergenceRadius=BestSolution(4);
    if text_verbose
        Parameters
    end

    %BATCH WORK
    fid=fopen(outputFile,'w');

    RandIndicesCV=ceil((NumFolds+1)*rand(1,NumSamples));
    for i=1:NumFolds
        fprintf('Fold: %d/%d\n',i,NumFolds);
        if cross_validation
            TrainSamplesCV=Samples(:,RandIndicesCV~=i);
            TestSamplesCV=Samples(:,RandIndicesCV==i);
        else
            TrainSamplesCV=Samples;
            TestSamplesCV=Samples;
        end

        %Standard SOFM
        Model=TrainSOFM(TrainSamplesCV,Parameters,graphic_verbose);

        %Forbidden Region SOM (FRSOM)
        ModelFR=TrainFRSOFM(TrainSamplesCV,Parameters,S,graphic_verbose);

        %Comparison
        [Winners,Errors,TopologyError,OffendingSamples]=CompetitionSOFMMEX(Model,TestSamplesCV);
        [WinnersFR,ErrorsFR,TopologyErrorFR,OffendingSamplesFR]=CompetitionSOFMMEX(ModelFR,TestSamplesCV);
        MSE=mean(Errors);
        MSEFR=mean(ErrorsFR);

        if text_verbose
           fprintf('MSE: %f, MSEFR: %f\n',MSE,MSEFR);       
        end
    %     if graphic_verbose        
    %         scatter(reshape(Model.Prototypes(1,:,:),[1,RowsCols^2]),reshape(Model.Prototypes(2,:,:),[1,RowsCols^2]),'g');
    %         scatter(reshape(ModelFR.Prototypes(1,:,:),[1,RowsCols^2]),reshape(ModelFR.Prototypes(2,:,:),[1,RowsCols^2]),'r');
    %     end

        %save partial results to file 
        fprintf(fid,'%s,%s,%d,%d,%d,%d,%d,%s,%d,%f,%f,%f,%f,%f,%f\r\n',dataFile,barriersFile,optim_method,cross_validation,Parameters.NumRowsMap,...
        Parameters.NumColsMap,Parameters.NumSteps,Parameters.Topology,Parameters.Toroidal,...
        Parameters.InitialLearningRate,Parameters.MaxRadius,Parameters.ConvergenceLearningRate,...
        Parameters.ConvergenceRadius,MSE,MSEFR);
    end

    %BATCH WORK
    fclose(fid);
end



