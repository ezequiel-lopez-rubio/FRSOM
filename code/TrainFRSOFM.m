function [Model]=TrainFRSOFM(Samples,Parameters,S,graphic_verbose)
% Train a Kohonen's SOFM model, Forbidden Region version
% S is a PolSCENE structure (see documentation)

[Dimension,NumSamples]=size(Samples);

%FOR VIDEO
VIDEO=false;
if VIDEO
    v = VideoWriter('overheadFRSOM.avi');
    open(v);
end

% Inicialization
NumNeuro=Parameters.NumRowsMap*Parameters.NumColsMap;
Model.NumColsMap=Parameters.NumColsMap;
Model.NumRowsMap=Parameters.NumRowsMap;
Model.Dimension=Dimension;
Model.Prototypes=zeros(Dimension,Model.NumRowsMap,Model.NumColsMap);

%initialize outside of barriers!!!
% so we choose random samples
prototypes=Samples(:,randi([1,NumSamples],[1,NumNeuro]));
for i=1:Dimension
    Model.Prototypes(i,:,:)=reshape(prototypes(i,:,:),Model.NumColsMap,Model.NumRowsMap);
end

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
SquareDistances=zeros(NumNeuro);
Paths=cell(NumNeuro);
if graphic_verbose
    h=scatter(reshape(Model.Prototypes(1,:,:),1,[]),reshape(Model.Prototypes(2,:,:),1,[]),'r');
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
    
    %SquaredDistances=sum((repmat(MySample,1,NumNeuro)-Model.Prototypes(:,:)).^2,1);
    
    for i=1:Model.NumRowsMap
        for j=1:Model.NumColsMap
            k=(i-1)*Model.NumColsMap+j;
            [distance,path]=PolSCENEShortestPath2Points(S,Model.Prototypes(:,i,j)',MySample(:)',0);%0 for no perturbation
            SquaredDistances(k)=distance;
            Paths{k}=path;
        end
    end
    [Minimum NdxWinner]=min(SquaredDistances);
    Coef=LearningRate*exp(-DistTopol{NdxWinner}/(MyRadius^2));
    
    % Update the neurons
    %Model.Prototypes(:,:)=Coef.*repmat(MySample,1,NumNeuro)+(1-Coef).*Model.Prototypes(:,:);
     for i=1:Model.NumRowsMap
        for j=1:Model.NumColsMap
            k=(i-1)*Model.NumColsMap+j;
            coef=min([1,Coef(k)]);
            path=Paths{k};
            Model.Prototypes(:,i,j)=RecorreCamino(path,[Model.Prototypes(1,i,j),Model.Prototypes(2,i,j)],[MySample(1),MySample(2)],coef)';
        end
     end

    if graphic_verbose        
        delete(h); 
        h=scatter(reshape(Model.Prototypes(1,:,:),1,[]),reshape(Model.Prototypes(2,:,:),1,[]),'filled','r');
        drawnow;
    end
    %NdxStep
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
end

    
    
        
