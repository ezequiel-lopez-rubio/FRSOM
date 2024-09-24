function [Winners,Errors,TopologyError,OffendingSamples]=CompetitionSOFM(Model,Samples)

NumSamples=size(Samples,2);
NumNeuro=Model.NumRowsMap*Model.NumColsMap;
Prototypes=Model.Prototypes(:,:);
Winners=zeros(NumSamples,1);
Errors=zeros(NumSamples,1);
OffendingSamples=zeros(NumSamples,1);

NumTopologyErrors=0;
MSE=0;
for NdxSample=1:NumSamples   
    SquaredDistances=sum((repmat(Samples(:,NdxSample),1,NumNeuro)-Prototypes).^2,1);
    [Minimum NdxWinner]=min(SquaredDistances);
    Winners(NdxSample)=NdxWinner;
    Errors(NdxSample)=Minimum;
    SquaredDistances(NdxWinner)=inf;
    [Minimum2 NdxWinner2]=min(SquaredDistances);
    MyDistTopol=Model.DistTopol{NdxWinner}(NdxWinner2);
    if MyDistTopol>1.1
        NumTopologyErrors=NumTopologyErrors+1;
        OffendingSamples(NdxSample)=1;
    end
end

TopologyError=NumTopologyErrors/NumSamples;