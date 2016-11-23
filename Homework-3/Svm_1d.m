clear all;
clc;
addpath('C:\Users\ShwetaS\Desktop\ML\HW\ShwetaSood_2012164_Ass2\libsvm-3.20\matlab\');
load('model');
w = (model.sv_coef' * full(model.SVs)); %Computing summation: alphai*yi*xi st xi belongs to SV
rho=model.rho;
b=-rho;

%Training SVM for best value of C
 testimages = loadMNISTImages('t10k-images.idx3-ubyte');
 testlabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
 pos_6=find(testlabels==6);
 pos_8=find(testlabels==8);
 len=length(pos_6)+length(pos_8);
 testimages_a=zeros(784,len); %Images to operate for part a
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
 [predicted_label, acc, prob_estimates]=svmpredict(testlabels_a,testset,model);
 
 correct=zeros(10,784);
 incorrect=zeros(10,784);
 correct_decision_value=zeros(size(correct,1),1);
 incorrect_decision_value=zeros(size(incorrect,1),1);
 
 p=1;
 q=1;
 for i=1:length(testlabels_a)
     if(testlabels_a(i)==predicted_label(i))
         correct(p,:)=testset(i,:);
         p=p+1;
     else
         incorrect(q,:)=testset(i,:);
         q=q+1;
     end
 end
 %{
%OPTION 1
 
%distance from hyperplane=|decisionvalue|/|w|

for i=1:size(correct,1)
    row=correct(i,:);
    d_val=w*transpose(row)+b;
    correct_decision_value(i)=abs(d_val);
end
for i=1:size(incorrect,1)
    row=incorrect(i,:);
    d_val=w*transpose(row)+b;
    incorrect_decision_value(i)=abs(d_val);
end
abs_w=sqrt(w*transpose(w));
plot_correct=abs(correct_decision_value)/abs_w;
plot_incorrect=abs(incorrect_decision_value)/abs_w;

one_1=zeros(size(plot_correct,1),1);
one_2=zeros(size(plot_incorrect,1),1);

for i=1:size(plot_correct,1)
    one_1(i)=i;
end
for i=1:size(plot_incorrect,1)
    one_2(i)=i;
end
%}
 
 
 %OPTION 2
 plot_correct=zeros(10,1);
 plot_incorrect=zeros(10,1);
 p=1;
 q=1;
  for i=1:length(testlabels_a)
     if(testlabels_a(i)==predicted_label(i))
         plot_correct(p)=abs(prob_estimates(i));
         p=p+1;
     else
         plot_incorrect(q)=abs(prob_estimates(i));
         q=q+1;
     end
 end
 
one_1=zeros(size(plot_correct,1),1);
one_2=zeros(size(plot_incorrect,1),1);

for i=1:size(plot_correct,1)
    one_1(i)=i;
end
for i=1:size(plot_incorrect,1)
    one_2(i)=i;
end


figure;
scatter(one_1,plot_correct,'filled','MarkerFaceColor','blue');
hold on;
scatter(one_2,plot_incorrect,'filled','MarkerFaceColor','red');
legend('Correct Classifications','Incorrect Classifications');
xlabel('Points');
ylabel('Distance');
title('Distance of correctly classified vs incorrectly classified points');
