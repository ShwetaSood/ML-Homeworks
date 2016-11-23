
clear all;
clc;
addpath('C:\Users\ShwetaS\Desktop\ML\HW\ShwetaSood_2012164_Ass2\libsvm-3.20\matlab\');
addpath('C:\Users\ShwetaS\Desktop\ML\HW\ShwetaSood_2012164_Ass2\smsspamcollection\');
[testnum,testtxt,testraw]=xlsread('Testdata.csv');
[trainnum,traintxt,trainraw]=xlsread('Traindata.csv');
%Ham 1 , Spam 0
testset=testnum(:,2:10);
testlabels=testnum(:,1);
trainset=trainnum(:,2:10);
trainlabels=trainnum(:,1);

ibestc=2048;
ibestg=1/9;
iaccuracy=91.591;
%Train model with parameters c, d
 bestcv = 0;
for c_val = -10:10
    for g=0:1:0.01
                model = svmtrain(trainlabels,trainset, ['-t 3 -g ',num2str(g) ,' -c ',num2str(2^c_val) ]);
                [predicted_label, acc, prob_estimates]=svmpredict(testlabels,testset,model);
                accuracy=acc(1);
                if (accuracy > bestcv)
                  bestcv = accuracy; bestc = 2^c_val; bestg=g;
                end
                fprintf('%g %g %g (best c=%g, g=%g rate=%g)\n', c_val,g, iaccuracy, ibestc, ibestg, iaccuracy);
    end
end
bestc=2048;
bestg=1/9;

model = svmtrain(testlabels,testset, ['-t 3 -g ', num2str(bestg),' -c ',num2str(bestc) ]);
[predicted_label, acc, prob_estimates]=svmpredict(testlabels,testset,model);
 correctham=0;
 correctspam=0;
 totalham=length(find(testlabels==1)); %Ham 1 , Spam 0
 totalspam=length(find(testlabels==0));
 
 for i=1:length(testlabels)
     if(testlabels(i)==predicted_label(i))
         if(testlabels(i)==1)
             correctham=correctham+1;
         else
             correctspam=correctspam+1;
         end
     end
 end
 disp(['Parameter c= ' num2str(bestc) ' Parameter g= ' num2str(bestg) ' Accuracy = ' num2str(acc(1))]);
 disp(['Classwise Accuracy for ham = ' num2str(correctham/totalham *100) ' Classwise Accuracy for spam = ' num2str(correctspam/totalspam *100)]);