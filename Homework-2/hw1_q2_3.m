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
        training(i,j)={num2str(tmp_data.data(i-1,j-1))}; % str2num(cell2mat(train(2,2)))
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


setdemorandstream(372880951)
net = patternnet([4,4]);
%view(net)
metraining=transpose(tmp_data.data);
label=zeros(7,size(training,1)-1);
    for j=1:size(training,1)-1
        if(strcmp(training(j+1,1),'BRICKFACE'))
            label(1,j)=1;
            label(2,j)=0;
            label(3,j)=0;
            label(4,j)=0;
            label(5,j)=0;
            label(6,j)=0;
            label(7,j)=0;
        elseif (strcmp(training(j+1,1),'SKY'))
            label(1,j)=0;
            label(2,j)=1;
            label(3,j)=0;
            label(4,j)=0;
            label(5,j)=0;
            label(6,j)=0;
            label(7,j)=0;
        elseif (strcmp(training(j+1,1),'FOLIAGE'))
            label(1,j)=0;
            label(2,j)=0;
            label(3,j)=1;
            label(4,j)=0;
            label(5,j)=0;
            label(6,j)=0;
            label(7,j)=0;
        elseif (strcmp(training(j+1,1),'CEMENT'))
            label(1,j)=0;
            label(2,j)=0;
            label(3,j)=0;
            label(4,j)=1;
            label(5,j)=0;
            label(6,j)=0;
            label(7,j)=0;
        elseif (strcmp(training(j+1,1),'WINDOW'))
            label(1,j)=0;
            label(2,j)=0;
            label(3,j)=0;
            label(4,j)=0;
            label(5,j)=1;
            label(6,j)=0;
            label(7,j)=0;
        elseif (strcmp(training(j+1,1),'PATH'))
            label(1,j)=0;
            label(2,j)=0;
            label(3,j)=0;
            label(4,j)=0;
            label(5,j)=0;
            label(6,j)=1;
            label(7,j)=0;
        elseif (strcmp(training(j+1,1),'GRASS'))
            label(1,j)=0;
            label(2,j)=0;
            label(3,j)=0;
            label(4,j)=0;
            label(5,j)=0;
            label(6,j)=0;
            label(7,j)=1;
        end
    end
    
metest=transpose(testtmp_data.data);
    
[net,tr] = train(net,metraining,label);
testY = net(metest);
testClasses = testY > 0.5;

label2=zeros(7,size(test,1)-1);
 for j=1:size(test,1)-1
        if(strcmp(test(j+1,1),'BRICKFACE'))
            label2(1,j)=1;
            label2(2,j)=0;
            label2(3,j)=0;
            label2(4,j)=0;
            label2(5,j)=0;
            label2(6,j)=0;
            label2(7,j)=0;
        elseif (strcmp(test(j+1,1),'SKY'))
            label2(1,j)=0;
            label2(2,j)=1;
            label2(3,j)=0;
            label2(4,j)=0;
            label2(5,j)=0;
            label2(6,j)=0;
            label2(7,j)=0;
        elseif (strcmp(test(j+1,1),'FOLIAGE'))
            label2(1,j)=0;
            label2(2,j)=0;
            label2(3,j)=1;
            label2(4,j)=0;
            label2(5,j)=0;
            label2(6,j)=0;
            label2(7,j)=0;
        elseif (strcmp(test(j+1,1),'CEMENT'))
            label2(1,j)=0;
            label2(2,j)=0;
            label2(3,j)=0;
            label2(4,j)=1;
            label2(5,j)=0;
            label2(6,j)=0;
            label2(7,j)=0;
        elseif (strcmp(test(j+1,1),'WINDOW'))
            label2(1,j)=0;
            label2(2,j)=0;
            label2(3,j)=0;
            label2(4,j)=0;
            label2(5,j)=1;
            label2(6,j)=0;
            label2(7,j)=0;
        elseif (strcmp(test(j+1,1),'PATH'))
            label2(1,j)=0;
            label2(2,j)=0;
            label2(3,j)=0;
            label2(4,j)=0;
            label2(5,j)=0;
            label2(6,j)=1;
            label2(7,j)=0;
        elseif (strcmp(test(j+1,1),'GRASS'))
            label2(1,j)=0;
            label2(2,j)=0;
            label2(3,j)=0;
            label2(4,j)=0;
            label2(5,j)=0;
            label2(6,j)=0;
            label2(7,j)=1;
        end
 end
 figure;
 plotconfusion(label2,testY)  
 title('Confusion matrix for Complete classification problem using double hidden layer network with 4 nodes each');
 total=0;
 cnt=0;
 msum=0;
 for i=1:size(label2,1)
     total=0;
     cnt=0;
      for j=1:size(label2,2)
          if(label2(i,j)==1)
              total=total+1;
              if(testClasses(i,j)==1)
                  cnt=cnt+1;
              end
          end
      end
      msum=msum+double(cnt/total);
 end
 b=msum/7*100;
 disp(['Mean Class-Wise Accuracy of the model is : ' num2str(b) '%']);