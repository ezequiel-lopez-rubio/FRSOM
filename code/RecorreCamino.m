function [ FinalPos ] = RecorreCamino( path,P1,P2,coef)
%RecorreCamino calcula coef% del camino contenido en path
%path debe ser una sucesión de puntos de R^n
%coef debe ser un número en [0,1]
%path debería empezar en P1 y acabar en P2. Los parámetros P1 y P2 se pasan
%por la siguiente casuística: si P1 y P2 no son alcanzables en la escena,
%entonces PolSCENEShortestPath2Points devuelve path=[].

Dimension=size(path,2);
NumVertices=size(path,1);

if NumVertices>=2 %otherwise this is not a proper path between two points
    %compute distances
    PartialDistances=zeros(1,NumVertices-1);
    AccumulatedDistances=zeros(1,NumVertices);
    AccumulatedDistance=0;
    AccumulatedDistances(1)=0;
    for i=1:NumVertices-1   
        PartialDistance=sqrt(sum((path(i+1,:)-path(i,:)).^2));
        PartialDistances(i)=PartialDistance;
        AccumulatedDistance=AccumulatedDistance+PartialDistance;
        AccumulatedDistances(i+1)=AccumulatedDistance;
    end
    TotalDistance=AccumulatedDistances(NumVertices);    
    ProportionalDistances=AccumulatedDistances/TotalDistance;
    PostVertex=find(ProportionalDistances>=coef,1);
    if PostVertex>=2
        PreVertex=PostVertex-1;
        PostVertexPos=path(PostVertex,:);
        PreVertexPos=path(PreVertex,:);
        FinalPos=PreVertexPos+(coef-ProportionalDistances(PreVertex))/(ProportionalDistances(PostVertex)-ProportionalDistances(PreVertex))*(PostVertexPos-PreVertexPos);
    else 
        FinalPos=P1;
    end
else
    FinalPos=P1;
end


