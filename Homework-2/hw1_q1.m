clear all;
clc;

%Creating training data
tmp_data=importdata('segmentation.data.txt',',');
train=tmp_data.textdata(4:size(tmp_data.textdata,1));
arr=strsplit(cell2mat(train(1,1)),',');
train(1,2:length(arr)+1)=arr;
train(1,1)={'CLASS'};
for i=2:2101
for j=2:20
    train(i,j)={num2str(tmp_data.data(i-1,j-1))}; % str2num(cell2mat(train(2,2)))
end
end

%Creating testing data
testtmp_data=importdata('segmentation.test.txt',',');
test=testtmp_data.textdata(4:size(testtmp_data.textdata,1));
arr=strsplit(cell2mat(test(1,1)),',');
test(1,2:length(arr)+1)=arr;
test(1,1)={'CLASS'};
for i=2:211
for j=2:20
    test(i,j)={num2str(testtmp_data.data(i-1,j-1))}; % str2num(cell2mat(train(2,2)))
end
end

%Classification
loop=5;
k=1;
for loop=5:10:105
    ensemble = TreeBagger(loop,tmp_data.data,train(2:2101));
    predictclass=ensemble.predict(testtmp_data.data);
    predictclass(:,2)=test(2:211,1); %1st column is predicted values, 2nd column is ground truth
    cnt=0;
    for i=1:210
        if(strcmp(predictclass(i,1),predictclass(i,2)))
             %count = sum(strcmp(predictclass(:,2),{'SKY'}),2);
             %sum(count==1);
             cnt=cnt+1;
        end
    end
    a=double(cnt/210)*100;

    disp(['Absolute Classification Accuracy of the model is : ' num2str(a)]);
    mat={'BRICKFACE','SKY','FOLIAGE','CEMENT','WINDOW','PATH','GRASS'};
    msum=0;
    for j=1:7
        count1 = sum(strcmp(predictclass(:,2),mat(j)),2);
        count2 = sum(strcmp(predictclass(:,1),mat(j)),2);
        cnt=0;
        total=0;
        for i=1:length(count1)
            if(count1(i)==1)
                total=total+1;
                if(count2(i)==1)
                    cnt=cnt+1;
                end
            end
        end
        msum=msum+(double(cnt/total));
    end
    b=msum/7*100;
    disp(['Mean Class-Wise Accuracy of the model is : ' num2str(b)]);
    arrabs(k)=a;
    arrmean(k)=b;
    k=k+1;
end

%Plotting
X=[5,15,25,35,45,55,65,75,85,95,105];
figure;
plot(X,arrabs);
xlabel('Number of Trees');
ylabel('Absolute Classification Accuracy');
ax = gca;
ax.XTick = [0,5,15,25,35,45,55,65,75,85,95,105];

figure;
plot(X,arrmean);
xlabel('Number of Trees');
ylabel('Mean Class-Wise Accuracy');
ax = gca;
ax.XTick = [0,5,15,25,35,45,55,65,75,85,95,105];