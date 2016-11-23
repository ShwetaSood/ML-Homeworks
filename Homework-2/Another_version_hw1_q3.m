clear all;
clc;

%Creating training data
tmp_data=importdata('segmentation.data.txt',',');
training=tmp_data.textdata(4:size(tmp_data.textdata,1));
arr=strsplit(cell2mat(training(1,1)),',');
training(1,2:length(arr)+1)=arr;
training(1,1)={'CLASS'};
for i=2:2101
    for j=2:20
        training(i,j)={num2str(tmp_data.data(i-1,j-1))};
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
        test(i,j)={num2str(testtmp_data.data(i-1,j-1))};
    end
end

Y=training(2:size(training,1),1);
t=classregtree(tmp_data.data,Y);
%view(t)
yfit=eval(t,testtmp_data.data);
yfit(:,2)=test(2:size(test,1),1);

%Classification
cnt=0;
    for i=1:210
        if(strcmp(yfit(i,1),yfit(i,2)))
             cnt=cnt+1;
        end
    end
    a=double(cnt/210)*100;
    disp(['Absolute Classification Accuracy of the model is : ' num2str(a)]);
    mat={'BRICKFACE','SKY','FOLIAGE','CEMENT','WINDOW','PATH','GRASS'};
    msum=0;
    for j=1:7
        count1 = sum(strcmp(yfit(:,2),mat(j)),2);
        count2 = sum(strcmp(yfit(:,1),mat(j)),2);
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