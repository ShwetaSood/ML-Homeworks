clear all;
clc;
load('model_rbf');

addpath('C:\Users\ShwetaS\Desktop\ML\HW\ShwetaSood_2012164_Ass2\libsvm-3.20\matlab\');
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
 %{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Q1 Part (a)
 pos_6=find(labels==6);
 pos_8=find(labels==8);
 len=length(pos_6)+length(pos_8);
 images_a=zeros(784,len); %Images to operate for part a
 labels_a=zeros(len,1);
 
 images_a1=zeros(784,5884); %Images to operate for part a
 labels_a1=zeros(5884,1);
 images_a2=zeros(784,5885); %Images to operate for part a
 labels_a2=zeros(5885,1);
 
 k=1;
 p=1;
 q=1;
 for i=1:length(pos_6)
     pos=pos_6(i);
     images_a(:,k)=images(:,pos);
     labels_a(k)=labels(pos);
     if(i<=length(pos_6)/2)
         images_a1(:,p)=images(:,pos);
         labels_a1(p)=labels(pos);
         p=p+1;
     else
         images_a2(:,q)=images(:,pos);
         labels_a2(q)=labels(pos);
         q=q+1;
     end
     k=k+1;
    
 end
 for i=1:length(pos_8)
     pos=pos_8(i);
     images_a(:,k)=images(:,pos);
     labels_a(k)=labels(pos);
     if(i<=length(pos_8)/2)
         images_a1(:,p)=images(:,pos);
         labels_a1(p)=labels(pos);
         p=p+1;
     else
         images_a2(:,q)=images(:,pos);
         labels_a2(q)=labels(pos);
         q=q+1;
     end
     k=k+1;
 end
 
 set=transpose(images_a);
 settrain=transpose(images_a1);
 settest=transpose(images_a2);
 
 bestcv = 0;
 accuracy=zeros(1,5);
 cval=zeros(1,5);
 k=1;
for c_val = -1:3
    for g_val=-4:1
        model = svmtrain(labels_a1,settrain, ['-t 2 -c ', num2str(2^c_val), ' -g ', num2str(2^g_val) ]);
        [predicted_label, acc, prob_estimates]=svmpredict(labels_a2,settest,model);
        cval(k)=2^c_val;
        accuracy(k)=acc(1);
        if (accuracy(k) > bestcv)
          bestcv = accuracy(k); bestc = 2^c_val; bestg=2^g_val;
        end
        fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', c_val, g_val, accuracy(k), bestc, bestg, bestcv);
        k=k+1;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Q1 Part (e)
%}
bestc=4; bestg=0.0625;
%Training SVM for best value of C
 testimages = loadMNISTImages('t10k-images.idx3-ubyte');
 testlabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
 pos_6=find(testlabels==6);
 pos_8=find(testlabels==8);
 len=length(pos_6)+length(pos_8);
 testimages_a=zeros(784,len);
 testlabels_a=zeros(len,1);
 k=1;
 for i=1:length(pos_6)
     pos=pos_6(i);
     testimages_a(:,k)=testimages(:,pos);
     testlabels_a(k)=testlabels(pos);
     k=k+1;
    
 end
 for i=1:length(pos_8)
     pos=pos_8(i);
     testimages_a(:,k)=testimages(:,pos);
     testlabels_a(k)=testlabels(pos);
     k=k+1;
 end
 
 testset=transpose(testimages_a);
 
% model = svmtrain(labels_a,set, ['-t 2 -c ', num2str(bestc), ' -g ', num2str(bestg) ]);
 [predicted_label, acc, prob_estimates]=svmpredict(testlabels_a,testset,model);
 correct6=0;
 correct8=0;
 total6=length(pos_6);
 total8=length(pos_8);
 
 for i=1:length(testlabels_a)
     if(testlabels_a(i)==predicted_label(i))
         if(testlabels_a(i)==6)
             correct6=correct6+1;
         else
             correct8=correct8+1;
         end
     end
 end
 disp(['Parameter C= ' num2str(bestc) ' Parameter G= ' num2str(bestg) ' Accuracy = ' num2str(acc(1))]);
 disp(['Classwise Accuracy for 6 = ' num2str(correct6/total6 *100) ' Classwise Accuracy for 8 = ' num2str(correct8/total8 *100)]);
