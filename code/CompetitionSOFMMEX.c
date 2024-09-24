#include "mex.h"
#include <math.h>
#include <float.h>


/* 
Coded by Ezequiel López-Rubio, March 2013.

In order to compile this function, type the following at the Matlab prompt:
>> mex CompetitionSOFMMEX.c

[Winners,Errors,TopologyError,OffendingSamples,SampleTiedRanks]=CompetitionSOFMMEX(Model,Samples)


*/

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
	mxArray *DistTopol,*TiedRank;
	double *ptrSamples,*ptrDistTopol,*ptrMySample,*ptrPrototypes,*ptrTiedRank;
	double *ptrWinners,*ptrErrors,*ptrTopologyError,*ptrOffendingSamples,*ptrSampleTiedRanks;
	int Dimension,NumSamples,NumRowsMap,NumColsMap,NumNeuro;
	int NdxSample,NdxWinner,NdxWinner2,NdxNeuro,NdxDim,NumTopologyErrors;
	double *ptrSquaredDistances;
	double MyDifference,MinSquaredDistance;


	/* Obtain working variables */
	ptrPrototypes=mxGetPr(mxGetField(prhs[0],0,"Prototypes"));
	DistTopol=mxGetField(prhs[0],0,"DistTopol");
	TiedRank=mxGetField(prhs[0],0,"TiedRank");
	ptrSamples=mxGetPr(prhs[1]);
	Dimension=(int)mxGetM(prhs[1]);
	NumSamples=(int)mxGetN(prhs[1]);
	NumRowsMap=(int)mxGetScalar(mxGetField(prhs[0],0,"NumRowsMap"));
	NumColsMap=(int)mxGetScalar(mxGetField(prhs[0],0,"NumColsMap"));
	NumNeuro=NumRowsMap*NumColsMap;

	/* Allocate dynamic memory */
	ptrSquaredDistances=mxMalloc(NumNeuro*sizeof(double));

	/* Create output mxArrays */
    plhs[0]=mxCreateDoubleMatrix(NumSamples,1,mxREAL);
	ptrWinners=mxGetPr(plhs[0]);
    plhs[1]=mxCreateDoubleMatrix(NumSamples,1,mxREAL);
	ptrErrors=mxGetPr(plhs[1]);
    plhs[2]=mxCreateDoubleMatrix(1,1,mxREAL);
	ptrTopologyError=mxGetPr(plhs[2]);
    plhs[3]=mxCreateDoubleMatrix(NumSamples,1,mxREAL);
	ptrOffendingSamples=mxGetPr(plhs[3]);
	plhs[4]=mxCreateDoubleMatrix(NumSamples,1,mxREAL);
	ptrSampleTiedRanks=mxGetPr(plhs[4]);

	/* Obtain topological distances and tied ranks */
	ptrDistTopol=mxMalloc(NumNeuro*NumNeuro*sizeof(double));
	ptrTiedRank=mxMalloc(NumNeuro*NumNeuro*sizeof(double));
	for(NdxNeuro=0;NdxNeuro<NumNeuro;NdxNeuro++)
	{
		memcpy(ptrDistTopol+NdxNeuro*NumNeuro,
			mxGetPr(mxGetCell(mxGetField(prhs[0],0,"DistTopol"),NdxNeuro)),
			NumNeuro*sizeof(double));
		memcpy(ptrTiedRank+NdxNeuro*NumNeuro,
			mxGetPr(mxGetCell(mxGetField(prhs[0],0,"TiedRank"),NdxNeuro)),
			NumNeuro*sizeof(double));
	}

	/* Process all training samples */
	NumTopologyErrors=0;
	for(NdxSample=0;NdxSample<NumSamples;NdxSample++)
	{
		/* Choose a training sample at random */
		ptrMySample=ptrSamples+NdxSample*Dimension;

		/* Find the smallest squared Euclidean distance from the sample to the prototypes */
		MinSquaredDistance=DBL_MAX;
		NdxWinner=0;
		for(NdxNeuro=0;NdxNeuro<NumNeuro;NdxNeuro++)
		{
			ptrSquaredDistances[NdxNeuro]=0.0;
			for(NdxDim=0;NdxDim<Dimension;NdxDim++)
			{
				MyDifference=ptrMySample[NdxDim]-ptrPrototypes[NdxDim+NdxNeuro*Dimension];
				ptrSquaredDistances[NdxNeuro]+=(MyDifference*MyDifference);
			}
			if (ptrSquaredDistances[NdxNeuro]<MinSquaredDistance)
			{
				MinSquaredDistance=ptrSquaredDistances[NdxNeuro];
				NdxWinner=NdxNeuro;
			}
		}

		/* Set the output matrices */
		ptrErrors[NdxSample]=MinSquaredDistance;
		ptrWinners[NdxSample]=NdxWinner+1;

		/* Set the distance of the winner to a large value */
		ptrSquaredDistances[NdxWinner]=DBL_MAX;

		/* Find the second smallest squared Euclidean distance from the sample to the prototypes */
		MinSquaredDistance=DBL_MAX;
		NdxWinner2=0;
		for(NdxNeuro=0;NdxNeuro<NumNeuro;NdxNeuro++)
		{
			if (ptrSquaredDistances[NdxNeuro]<MinSquaredDistance)
			{
				MinSquaredDistance=ptrSquaredDistances[NdxNeuro];
				NdxWinner2=NdxNeuro;
			}
		}

		/* Check whether a topology error has occurred */
		if (ptrDistTopol[NdxWinner2+NdxWinner*NumNeuro]>1.1)
		{
			ptrOffendingSamples[NdxSample]=1.0;
			NumTopologyErrors++;
		}

		/* Obtain tied rank for the sample */
		ptrSampleTiedRanks[NdxSample]=ptrTiedRank[NdxWinner2+NdxWinner*NumNeuro];

	}

	/* Set the resulting topology error */
	(*ptrTopologyError)=(double)NumTopologyErrors/(double)NumSamples;


	/* Release dynamic memory */
	mxFree(ptrDistTopol);
	mxFree(ptrTiedRank);
	mxFree(ptrSquaredDistances);


}