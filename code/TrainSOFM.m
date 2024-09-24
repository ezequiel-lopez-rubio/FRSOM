function [Model]=TrainSOFM(Samples,Parameters,graphic_verbose)
% Train a Kohonen's SOFM model, standard version
[Dimension,NumSamples]=size(Samples);

%FOR VIDEO
VIDEO=false;
if VIDEO
    v = VideoWriter('overheadSOM.avi');
    open(v);
end
    
% Inicialization
NumNeuro=Parameters.NumRowsMap*Parameters.NumColsMap;
Model.NumColsMap=Parameters.NumColsMap;
Model.NumRowsMap=Parameters.NumRowsMap;
Model.Dimension=Dimension;
Model.Prototypes=zeros(Dimension,Model.NumRowsMap,Model.NumColsMap);


% we choose random samples
prototypes=Samples(:,randi([1,NumSamples],[1,NumNeuro]));
for i=1:Dimension
    Model.Prototypes(i,:,:)=reshape(prototypes(i,:,:),Model.NumColsMap,Model.NumRowsMap);
end

% % Initialize along the two first principal directions
% Options.disp=0;
% Mu=mean(Samples,2);
% if NumSamples>Dimension
%     C=cov(Samples');
%     if Dimension>3
%         Model.GlobalMu=Mu;
%         [Uq Lambdaq]=eigs(C,3,'LM',Options);
%         Model.UqT=Uq';
%     end
%     [Uq Lambdaq]=eigs(C,2,'LM',Options);    
% else
%     % We use the eigenface trick here
%     SamplesZeroMean=Samples-repmat(Mu,1,NumSamples); 
%     L=SamplesZeroMean'*SamplesZeroMean;
%     [Lvectors Lvalues]=eigs(L,3,'LM',Options);
%     Uq=normc(SamplesZeroMean*Lvectors);
%     Model.UqT=Uq';
%     Model.GlobalMu=Mu;
%     Lambdaq=Lvalues/(NumSamples-1);  
%     % Next we only need the two first principal directions
%     Uq=Uq(:,1:2);
%     Lambdaq=Lambdaq(1:2,1:2);
% end
% UqLambdaq=Uq*sqrt(Lambdaq);
% A=zeros(2,1);
% for NdxRow=1:Model.NumRowsMap
%     A(1)=-0.5+NdxRow/Model.NumRowsMap;
%     for NdxCol=1:Model.NumColsMap
%         A(2)=-0.5+NdxCol/Model.NumColsMap;
%         Model.Prototypes(:,NdxRow,NdxCol)=Mu+UqLambdaq*A;  
%     end
% end

switch Parameters.Topology
    case 'Square'
        [NeuronCoords,DistTopol]=CreateSquareGrid(Parameters.NumRowsMap,Parameters.NumColsMap,Parameters.Toroidal);
    case 'Hex'
        [NeuronCoords,DistTopol]=CreateHexGrid(Parameters.NumRowsMap,Parameters.NumColsMap,Parameters.Toroidal);
    case 'Tri'
        [NeuronCoords,DistTopol]=CreateTriGrid(Parameters.NumRowsMap,Parameters.NumColsMap,Parameters.Toroidal);
    case 'Cairo'
        [NeuronCoords,DistTopol]=CreateCairoGrid(Parameters.NumRowsMap,Parameters.NumColsMap,Parameters.Toroidal);
    case 'Prismatic'
        [NeuronCoords,DistTopol]=CreatePrismaticGrid(Parameters.NumRowsMap,Parameters.NumColsMap,Parameters.Toroidal);
end
Model.NeuronCoords=NeuronCoords;
Model.DistTopol=DistTopol;
Model.TiedRank=cell(Model.NumRowsMap,Model.NumColsMap);
for NdxNeuron=1:Model.NumRowsMap*Model.NumColsMap
    Model.TiedRank{NdxNeuron}=reshape(tiedrank(Model.DistTopol{NdxNeuron}(:))-1,...
        [Model.NumRowsMap Model.NumColsMap]);
end
    

% Training
%Model=TrainSOFMMEX(Model,Samples,Parameters);
if graphic_verbose
    h=scatter(reshape(Model.Prototypes(1,:,:),1,[]),reshape(Model.Prototypes(2,:,:),1,[]),'g');
end
for NdxStep=1:Parameters.NumSteps
    MySample=Samples(:,ceil(NumSamples*rand(1)));
    if NdxStep<0.5*Parameters.NumSteps   
        % Ordering phase: linear decay
        LearningRate=Parameters.InitialLearningRate*(1-NdxStep/Parameters.NumSteps);
        MyRadius=Parameters.MaxRadius*(1-(NdxStep-1)/Parameters.NumSteps);
    else
        % Convergence phase: constant
        LearningRate=Parameters.ConvergenceLearningRate;
        MyRadius=Parameters.ConvergenceRadius;
    end
    
    SquaredDistances=sum((repmat(MySample,1,NumNeuro)-Model.Prototypes(:,:)).^2,1);
    [Minimum NdxWinner]=min(SquaredDistances);
    Coef=repmat(LearningRate*exp(-DistTopol{NdxWinner}/(MyRadius^2)),Dimension,1);

    % Update the neurons
    Model.Prototypes(:,:)=Coef.*repmat(MySample,1,NumNeuro)+...
        (1-Coef).*Model.Prototypes(:,:);

    if graphic_verbose
        delete(h); 
        h=scatter(reshape(Model.Prototypes(1,:,:),1,[]),reshape(Model.Prototypes(2,:,:),1,[]),'filled','g');
        drawnow;
    end
    %NdxStep
    
    %FOR VIDEO
    if VIDEO
       frame = getframe;
       writeVideo(v,frame);
    end
end


if graphic_verbose
    delete(h);
end

%FOR VIDEO
if VIDEO
    close(v);
    delete(h);
end


    
    
        
