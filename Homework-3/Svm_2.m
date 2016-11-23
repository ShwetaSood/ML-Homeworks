%Idea - the further a test vector is from the hyperplane the more it belongs to a certain class
%One-versus-all multiclass svm for the entire MNIST dataset (10 classes)

clear all;
clc;
addpath('C:\Users\ShwetaS\Desktop\ML\HW\ShwetaSood_2012164_Ass2\libsvm-3.20\matlab\');
%{
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

%t10k-images.idx3-ubyte
%t10k-labels.idx1-ubyte

%Building 10 base RBF classifiers with c=4, g=0.0625 (from experiments done in earlier parts)
testimages=transpose(images);
testlabels=zeros(length(labels),1);

%Training 10 base classifiers
c=4;
g=0.0625;
model=cell(10,1);
for i=0:9
  testlabels(labels==i,1)=1;
  model{i+1} = svmtrain(testlabels,testimages, ['-t 2 -c ', num2str(c), ' -g ', num2str(g) ]);
 %[predicted_label, acc, prob_estimates]=svmpredict(testlabels_a,testset,model);
 testlabels(labels==i,1)=0;
end
%}
%{
%Testing
load('matlab_341am');
 trainimages = loadMNISTImages('t10k-images.idx3-ubyte');
 trainimages=transpose(trainimages);
 trainlabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
 anslabel=ones(10000,1)*-1;
 correct=0;
 
 for i=1:length(trainimages)
     currmax=-1;othermax=100;
     f=-1;
     for j=1:10
        [predicted_label, acc, prob_estimates]=svmpredict(trainlabels(i,1),trainimages(i,:),model{j,1});
        if(abs(prob_estimates)>currmax && predicted_label==1) %if 1 is predicted we choose classifier with max confidence
            currmax=abs(prob_estimates);
            pos=j-1;
            f=1;
        elseif(f~=1 && abs(prob_estimates)<othermax) %if 1 is never predicted we choose classifier with lowest confidence
            othermax=abs(prob_estimates);
            otherpos=j-1;
        end
     end
     if(f==1)
        anslabel(i)=pos;
     else
         anslabel(i)=otherpos;
     end
     if(anslabel(i)==trainlabels(i))
         correct=correct+1;
     end
 end
%}
load('matlab_456pm');
disp(['Classification Accuracy is : ' , num2str(correct/100) '%']);
classcorrect=zeros(10,1);
total=zeros(10,1);
disp('Classwise Accuracy is :');
for i=1:length(trainlabels)
    if(trainlabels(i)==anslabel(i))
        pos=trainlabels(i);
        classcorrect(pos+1)=classcorrect(pos+1)+1;
    end
    total(trainlabels(i)+1)=total(trainlabels(i)+1)+1;
end
for i=1:10
    disp(['Class ' num2str(i-1) ' : ' num2str(classcorrect(i)/total(i)*100) '%']);
end
label2=zeros(10,length(trainlabels));
out2=zeros(10,length(trainlabels));
for i=1:length(trainlabels)
    pos=trainlabels(i);
    label2(pos+1,i)=1;
    pos2=anslabel(i);
    out2(pos2+1,i)=1;
end
figure;
plotconfusion(label2,out2)  
title('10-Class Confusion matrix');