function [NeuronCoords,DistTopol]=CreateSquareGrid(NumRows,NumCols,Toroidal)
% Square grid

NumNeurons=NumRows*NumCols;
BigNeuronCoords=zeros(2,2*NumRows,2*NumCols);
DistTopol=cell(NumRows,NumCols);

% Generate the neuron coords
for NdxRow=0:(2*NumRows-1)
    BigNeuronCoords(1,NdxRow+1,:)=NdxRow;
    for NdxCol=0:(2*NumCols-1)  
        BigNeuronCoords(2,NdxRow+1,NdxCol+1)=NdxCol;
    end
end

NeuronCoords=BigNeuronCoords(:,1:NumRows,1:NumCols);

if Toroidal
    for NdxRow=1:NumRows
        for NdxCol=1:NumCols
            NdxNeurons=sub2ind([2*NumRows 2*NumCols],[NdxRow NdxRow+NumRows NdxRow NdxRow+NumRows],...
                [NdxCol NdxCol NdxCol+NumCols NdxCol+NumCols]);
            DistTopol{NdxRow,NdxCol}=inf*ones(1,NumNeurons);
            for NdxTry=1:4
                DistTopol{NdxRow,NdxCol}=min(DistTopol{NdxRow,NdxCol},...
                    sqrt(sum((repmat(BigNeuronCoords(:,NdxNeurons(NdxTry)),1,NumNeurons)-NeuronCoords(:,:)).^2,1)));
            end
        end
    end
else
    for NdxRow=1:NumRows
        for NdxCol=1:NumCols
            NdxNeuro=sub2ind([NumRows NumCols],NdxRow,NdxCol);
            DistTopol{NdxRow,NdxCol}=sqrt(sum((repmat(NeuronCoords(:,NdxNeuro),1,NumNeurons)-NeuronCoords(:,:)).^2,1));
        end
    end
end

