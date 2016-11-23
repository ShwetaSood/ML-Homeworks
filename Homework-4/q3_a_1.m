clear all;
clc;

%Creating training and testing data as per specifications
tmp_data=importdata('wine.data.txt',',');
rows=size(tmp_data,1);
cols=size(tmp_data,2);
c1=find(tmp_data(:,1)==1);
c2=find(tmp_data(:,1)==2);
c3=find(tmp_data(:,1)==3);
train=vertcat(tmp_data(c1(1:floor(length(c1)/2)),:),tmp_data(c2(1:floor(length(c2)/2)),:),tmp_data(c3(1:floor(length(c3)/2)),:));
test=vertcat(tmp_data(c1(floor(length(c1)/2+1):length(c1)),:),tmp_data(c2(floor(length(c2)/2+1):length(c2)),:),tmp_data(c3(floor(length(c3)/2+1):length(c3)),:));
train_labels=train(:,1);
train=train(:,2:14);
test_labels=test(:,1);
test=test(:,2:14);

%Applying K-means
%Initializing K randome cluster centroids on parameters mean, mode, median
%Setting K=3
centroid=zeros(3,13);
for i=1:13
    centroid(1,i)=mean(train(:,i));
    centroid(2,i)=median(train(:,i));
    centroid(3,i)=mode(train(:,i));
end

%using Euclidean
prev_sse=inf;
iter1=0;
m=1;
while(1)
    new_sse=0;
    p=1;q=1;r=1;
    cluster1=zeros(1,13);
    cluster2=zeros(1,13);
    cluster3=zeros(1,13);
    class1=zeros(1,1);
    class2=zeros(1,1);
    class3=zeros(1,1);
    for i=1:size(train,1)
        x=train(i,:);
        dis1=euclidean(x,centroid(1,:));
        dis2=euclidean(x,centroid(2,:));
        dis3=euclidean(x,centroid(3,:));
        %{
        v1(m,1)=dis1;
        v1(m,2)=dis2;
        v1(m,3)=dis3;
        v2(m,1)=chebyshev(x,centroid(1,:));
        v2(m,2)=chebyshev(x,centroid(2,:));
        v2(m,3)=chebyshev(x,centroid(3,:));
        v3(m,1)=v1(m,1)-v2(m,1);
        v3(m,2)=v1(m,2)-v2(m,2);
        v3(m,3)=v1(m,3)-v2(m,3);
        m=m+1;
        %}
        if(dis1<=dis2)
            if(dis1<=dis3)
                cluster1(p,:)=train(i,:);
                class1(p)=i;
                p=p+1;
                new_sse=new_sse+(dis1^2);
            else
                cluster3(r,:)=train(i,:);
                class3(r)=i;
                r=r+1;
                new_sse=new_sse+(dis3^2);
            end
        else
            if(dis2<=dis3)
                cluster2(q,:)=train(i,:);
                class2(q)=i;
                q=q+1;
                new_sse=new_sse+(dis2^2);
            else
                cluster3(r,:)=train(i,:);
                class3(r)=i;
                r=r+1;
                new_sse=new_sse+(dis3^2);
            end
        end
    end

    for k=1:13
        centroid(1,k)=mean(cluster1(:,k));
        centroid(2,k)=mean(cluster2(:,k));
        centroid(3,k)=mean(cluster3(:,k));
    end
    if(prev_sse~=new_sse)
        prev_sse=new_sse;
    else
        break;
    end
    iter1=iter1+1;
end

%3(b) Assignning each cluster a class label - Euclidean
ach_e=zeros(3,size(train,1));
obt_e=zeros(3,size(train,1));

class1_given=zeros(length(class1),1);
class2_given=zeros(length(class2),1);
class3_given=zeros(length(class3),1);

for i=1:length(class1)
    point=class1(i);
    class1_given(i)=train_labels(point);
end
for i=1:length(class2)
    point=class2(i);
    class2_given(i)=train_labels(point);
end
for i=1:length(class3)
    point=class3(i);
    class3_given(i)=train_labels(point);
end
class1_label_e=mode(class1_given);
class2_label_e=mode(class2_given);
class3_label_e=mode(class3_given);
acc_1=0;
acc_2=0;
acc_3=0;
col=1;
for i=1:length(class1)
    point=class1(i);
    var=train_labels(point);
    ach_e(var,col)=1;
    obt_e(class1_label_e,col)=1;
    if(var==class1_label_e)
        acc_1=acc_1+1;
    end
    col=col+1;
end
for i=1:length(class2)
    point=class2(i);
    var=train_labels(point);
    ach_e(var,col)=1;
    obt_e(class2_label_e,col)=1;
    if(var==class2_label_e)
        acc_2=acc_2+1;
    end
    col=col+1;
end
for i=1:length(class3)
    point=class3(i);
    var=train_labels(point);
    ach_e(var,col)=1;
    obt_e(class3_label_e,col)=1;
    if(var==class3_label_e)
        acc_3=acc_3+1;
    end
    col=col+1;
end

%3c Confusion matrix and training error
plotconfusion(ach_e,obt_e);
disp(['Training Classwise Accuracy is:']);
disp(['Class 1:',num2str(acc_1/29*100)]);
disp(['Class 2:',num2str(acc_3/35*100)]);
disp(['Class 3:',num2str(acc_2/24*100)]);
disp(['Overall Class Accuracy is :', num2str((acc_1+acc_2+acc_3)/(size(train,1))*100)]);