#include "mex.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>


/* 
Coded by Ezequiel López-Rubio, March 2013.

In order to compile this function, type the following at the Matlab prompt:
>> mex TrainSOFMMEX.c

TrainedModel = TrainSOFMMEX(UntrainedModel,Samples,Parameters)


*/

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
	mxArray *DistTopol;
	double *ptrSamples,*ptrDistTopol,*ptrMySample,*ptrPrototypes,*ptrCoef;
	int Dimension,NumSamples,NumRowsMap,NumColsMap,NumNeuro;
	int NumSteps,NdxMySample,NdxWinner,NdxNeuro,NdxStep,NdxDim;
	double InitialLearningRate,MaxRadius,ConvergenceLearningRate,ConvergenceRadius;
	double LearningRate,MyRadius,MyRadius2,MinSquaredDistance;
	double ThisSquaredDistance,MyDifference;

	

	/* Initialize random seed */
	srand((unsigned int)time(NULL));

	/* Create output mxArray */
    plhs[0]=mxDuplicateArray(prhs[0]);

	/* Obtain working variables */
	ptrPrototypes=mxGetPr(mxGetField(plhs[0],0,"Prototypes"));
	DistTopol=mxGetField(plhs[0],0,"DistTopol");
	ptrSamples=mxGetPr(prhs[1]);
	Dimension=(int)mxGetM(prhs[1]);
	NumSamples=(int)mxGetN(prhs[1]);
	NumRowsMap=(int)mxGetScalar(mxGetField(plhs[0],0,"NumRowsMap"));
	NumColsMap=(int)mxGetScalar(mxGetField(plhs[0],0,"NumColsMap"));
	NumNeuro=NumRowsMap*NumColsMap;

	/* Allocate dynamic memory */
	ptrCoef=mxMalloc(NumNeuro*sizeof(double));
	
	/* Obtain traning parameters */
	NumSteps=(int)mxGetScalar(mxGetField(prhs[2],0,"NumSteps"));
	InitialLearningRate=mxGetScalar(mxGetField(prhs[2],0,"InitialLearningRate"));
	MaxRadius=mxGetScalar(mxGetField(prhs[2],0,"MaxRadius"));
	ConvergenceLearningRate=mxGetScalar(mxGetField(prhs[2],0,"ConvergenceLearningRate"));
	ConvergenceRadius=mxGetScalar(mxGetField(prhs[2],0,"ConvergenceRadius"));

	/* Obtain topological distances */
	ptrDistTopol=mxMalloc(NumNeuro*NumNeuro*sizeof(double));
	for(NdxNeuro=0;NdxNeuro<NumNeuro;NdxNeuro++)
	{
		memcpy(ptrDistTopol+NdxNeuro*NumNeuro,
			mxGetPr(mxGetCell(mxGetField(plhs[0],0,"DistTopol"),NdxNeuro)),
			NumNeuro*sizeof(double));
	}

	for(NdxStep=0;NdxStep<NumSteps;NdxStep++)
	{
		/* Choose a training sample at random */
		NdxMySample=rand()%NumSamples;
		ptrMySample=ptrSamples+NdxMySample*Dimension;

		/* Setup learning parameters */
		if (NdxStep<NumSteps/2)
		{
			/* Ordering phase: linear decay */
			LearningRate=InitialLearningRate*(1.0-(double)NdxStep/(double)NumSteps);
			MyRadius=MaxRadius*(1.0-(double)(NdxStep-1)/(double)NumSteps);
		}
		else
		{
			/* Convergence phase: constant */
			LearningRate=ConvergenceLearningRate;
			MyRadius=ConvergenceRadius;
		}
		MyRadius2=MyRadius*MyRadius;

		/* Find the smallest squared Euclidean distance from the sample to the prototypes */
		MinSquaredDistance=DBL_MAX;
		NdxWinner=0;
		for(NdxNeuro=0;NdxNeuro<NumNeuro;NdxNeuro++)
		{
			ThisSquaredDistance=0.0;
			for(NdxDim=0;NdxDim<Dimension;NdxDim++)
			{
				MyDifference=ptrMySample[NdxDim]-ptrPrototypes[NdxDim+NdxNeuro*Dimension];
				ThisSquaredDistance+=(MyDifference*MyDifference);
			}
			if (ThisSquaredDistance<MinSquaredDistance)
			{
				MinSquaredDistance=ThisSquaredDistance;
				NdxWinner=NdxNeuro;
			}
		}

		/* Compute update coefficients */
		for(NdxNeuro=0;NdxNeuro<NumNeuro;NdxNeuro++)
		{
			ptrCoef[NdxNeuro]=LearningRate*exp(-ptrDistTopol[NdxNeuro+NdxWinner*NumNeuro]/MyRadius2);
		}
		
		/* Update the prototypes */
		for(NdxNeuro=0;NdxNeuro<NumNeuro;NdxNeuro++)
		{
			for(NdxDim=0;NdxDim<Dimension;NdxDim++)
			{
				ptrPrototypes[NdxDim+NdxNeuro*Dimension]=
					ptrCoef[NdxNeuro]*ptrMySample[NdxDim]+
					(1.0-ptrCoef[NdxNeuro])*ptrPrototypes[NdxDim+NdxNeuro*Dimension];
			}
		}
	}

	/* Release dynamic memory */
	mxFree(ptrDistTopol);
	mxFree(ptrCoef);


}