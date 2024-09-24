function [Value]=ObjectiveFunction(x,Parameters,TrainSamples,ValidationSamples,MSE_Zero)

Parameters.InitialLearningRate=x(1);
Parameters.MaxRadius=x(2);
Parameters.ConvergenceLearningRate=x(3);
Parameters.ConvergenceRadius=x(4);

Model=TrainSOFM(TrainSamples,Parameters,0);
[~,Errors]=CompetitionSOFMMEX(Model,ValidationSamples);

MSE=mean(Errors);
Value=MSE/MSE_Zero;


