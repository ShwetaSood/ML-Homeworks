clear all;
clc;
vector=[150,165,150];
tmp_data=importdata('Skin_NonSkin.txt',' ');
%disp(tmp_data(:,4));
no_1=histc(tmp_data(:,4),1);
no_2=histc(tmp_data(:,4),2);
sample=(no_1)/3;
test=zeros(2*sample,4);
train=zeros(2*sample,4);
validate=zeros(2*sample,4);
train_new=zeros(4*sample,4);
 %Performing 3 fold crossvalidation
 
test(1:sample,:)=tmp_data(1:sample,:);
test(sample+1:2*sample,:)=tmp_data(no_1+1:no_1+sample,:);

train(1:sample,:)=tmp_data(sample+1:2*sample,:);
train(sample+1:2*sample,:)=tmp_data(no_1+1+sample:no_1+2*sample,:);

validate(1:sample,:)=tmp_data(2*sample+1:3*sample,:);
validate(sample+1:2*sample,:)=tmp_data(no_1+1+2*sample:no_1+3*sample,:);
%{
temp=test;
test=train;
train=temp;
%}

for loop=1:3
   
if(loop==2)
    orgtest=test;
    orgtrain=train;
    orgvalidate=validate;
    temp=test;
    test=validate;
    validate=temp;
end
 

 if(loop==3)
    temp=orgtest;
    test=orgtrain;
    train=temp;
    validate=orgvalidate;   
end   
    
train_new(1:sample,:)=train(1:sample,:);
train_new(sample+1:2*sample,:)=validate(1:sample,:);
train_new(2*sample+1:3*sample,:)=train(sample+1:2*sample,:);
train_new(3*sample+1:4*sample,:)=validate(sample+1:2*sample,:);

%Q4 1. Range, mean, variance of each attribute.

r1=range(train_new(1:2*sample,1:3));
m1=mean(train_new(1:2*sample,1:3));
v1=var(train_new(1:2*sample,1:3));
fprintf('Range for each attribute is: ');
disp(range(train_new(1:2*sample,1:3)));
fprintf('Mean for each attribute is: ');
disp(mean(train_new(1:2*sample,1:3)));
fprintf('Variance for each attribute is: ');
disp(var(train_new(1:2*sample,1:3)));


%Q4 2. Histogram for each attribute for two classes


f1=figure;
hist(train_new(1:2*sample,1));
hold on;
h = findobj(gca,'Type','patch');
set(h,'FaceColor','r','EdgeColor','w');
hist(train_new(2*sample+1:4*sample,1));
legend('Class 1','Class 2');
title('Histogram based on attribute 1 for the 2 classes');
hold off;
%saveas(f1,'Histo_A1.jpg') 

f2=figure;
hist(train_new(1:2*sample,2));
hold on;
h = findobj(gca,'Type','patch');
set(h,'FaceColor','r','EdgeColor','w');
hist(train_new(2*sample+1:4*sample,2));
legend('Class 1','Class 2');
title('Histogram based on attribute 2 for the 2 classes');
hold off;
%saveas(f2,'Histo_A2.jpg') 

f3=figure;
hist(train_new(1:2*sample,3));
hold on;
h = findobj(gca,'Type','patch');
set(h,'FaceColor','r','EdgeColor','w');
hist(train_new(2*sample+1:4*sample,3));
legend('Class 1','Class 2');
title('Histogram based on attribute 3 for the 2 classes');
tickVector=[0,50,100,150,200,250];
%tickVector=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250];
set(gca,'XTick',tickVector);
hold off;
%saveas(f3,'Histo_A3.jpg') 


%Q4 3. Analyze Histogram and select attribute having most discriminatory
%behavior

%Q4 4. Report TPR, FPR, TNR, FNR
%thres=150;
thres=vector(loop);
tp=0;
fp=0;
tn=0;
fn=0;
for i=1:2*sample
    if(test(i,3)>thres)
        if(test(i,4)==1)
            tp=tp+1;
        else
            fp=fp+1;
        end
    else
        if(test(i,4)==2)
            tn=tn+1;
        else
            fn=fn+1;
        end
    end
end

tpr=tp/(tp+fn);
fpr=fp/(fp+tn);
tnr=tn/(tn+fp);
fnr=fn/(fn+tp);
fprintf('TPR: ');
disp(tpr);
fprintf('FPR: ');
disp(fpr);
fprintf('TNR: ');
disp(tnr);
fprintf('FNR: ');
disp(fnr);


%Q4 5. Pick two attributes at a time and create a scatter plot with
%training data for two classes

f1=figure;
scatter(train_new(1:2*sample,1),train_new(1:2*sample,2),'r');
hold on;
scatter(train_new(2*sample+1:4*sample,1),train_new(2*sample+1:4*sample,2),'g');
legend('Class 1','Class 2');
title('Scatter plot based on attribute 1,2 for the 2 classes');

f2=figure;
scatter(train_new(1:2*sample,2),train_new(1:2*sample,3),'r');
hold on;
scatter(train_new(2*sample+1:4*sample,2),train_new(2*sample+1:4*sample,3),'g');
legend('Class 1','Class 2');
title('Scatter plot based on attribute 2,3 for the 2 classes');

f3=figure;
scatter(train_new(1:2*sample,1),train_new(1:2*sample,3),'r');
hold on;
scatter(train_new(2*sample+1:4*sample,1),train_new(2*sample+1:4*sample,3),'g');
legend('Class 1','Class 2');
title('Scatter plot based on attribute 1,3 for the 2 classes');


%Q4 6. Which pair has most discriminatory behavior for given 2 class
%problem

end