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
        training(i,j)={num2str(tmp_data.data(i-1,j-1))}; % str2num(cell2mat(training(2,2)))
    end
end
%training(not(strcmp(training(:,1),'SKY')),:)=[];
pos1=strcmp(training(:,1),'SKY');
pos2=strcmp(training(:,1),'BRICKFACE');
pos=pos1 | pos2;
training(not(pos),:)=[];
tmp_data.data(:,:)=[];
for i=1:size(training,1)
    for j=2:20
        tmp_data.data(i,j-1)=str2num(cell2mat(training(i,j)));
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
        test(i,j)={num2str(testtmp_data.data(i-1,j-1))}; % str2num(cell2mat(training(2,2)))
    end
end

pos1=strcmp(test(:,1),'SKY');
pos2=strcmp(test(:,1),'BRICKFACE');
pos=pos1 | pos2;
test(not(pos),:)=[];
testtmp_data.data(:,:)=[];
for i=1:size(test,1)
    for j=2:20
        testtmp_data.data(i,j-1)=str2num(cell2mat(test(i,j)));
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       PART B
%Creating double hidden layer network with 4 nodes each
%Loop for all points in train

weights1 = -1*2.*rand(19,4);
weights2 = -1*2.*rand(16,1);
weightsout = -1*2.*rand(4,1);

iter=0;
loop=0;
for loop=1:200
    for j=1:size(tmp_data.data,1)
        iter=0;

        a=perceptron2(weights1(1:19,1),tmp_data.data(j,:));
        b=perceptron2(weights1(1:19,2),tmp_data.data(j,:));
        c=perceptron2(weights1(1:19,3),tmp_data.data(j,:));
        d=perceptron2(weights1(1:19,4),tmp_data.data(j,:));
        

        e=perceptron2(weights2(1:4,1),[a,b,c,d]);
        f=perceptron2(weights2(5:8,1),[a,b,c,d]);
        g= perceptron2(weights2(9:12,1),[a,b,c,d]);
        h= perceptron2(weights2(13:16,1),[a,b,c,d]);

        out=perceptron2(weightsout(1:4,1),[e,f,g,h]);
        if(strcmp(test(j),'SKY')) %loop
                reqout(j,1)=0; %loop
            else
                reqout(j,1)=1; %loop
        end
        %BackPropagation of Error
        if(out>0)
            out=1;
        else
            out=0;
        end
        deltaout=reqout(j,1)-out;

        delta1=deltaout*weightsout(1,1);
        delta2=deltaout*weightsout(2,1);
        delta3=deltaout*weightsout(3,1);
        delta4=deltaout*weightsout(4,1);
        delta5=delta1*weights2(1,1)+delta2*weights2(5,1)+delta3*weights2(9,1)+delta4*weights2(13,1);
        delta6=delta1*weights2(2,1)+delta2*weights2(6,1)+delta3*weights2(10,1)+delta4*weights2(14,1);
        delta7=delta1*weights2(3,1)+delta2*weights2(7,1)+delta3*weights2(11,1)+delta4*weights2(15,1);
        delta8=delta1*weights2(4,1)+delta2*weights2(8,1)+delta3*weights2(12,1)+delta4*weights2(16,1);

        %Weight Updation
        coeff = 0.7; %eta
        
        for i=1:19
            weights1(i,1)=weights1(i,1)+coeff*delta5*tmp_data.data(j,i);
        end
        for i=1:19
            weights1(i,2)=weights1(i,2)+coeff*delta6*tmp_data.data(j,i);
        end
        for i=1:19
            weights1(i,3)=weights1(i,3)+coeff*delta7*tmp_data.data(j,i);
        end
        for i=1:19
            weights1(i,4)=weights1(i,4)+coeff*delta8*tmp_data.data(j,i);
        end
 

        for i=1:4
            weights2(i,1)=weights2(i,1)+coeff*delta1*a;
        end
        for i=5:8
            weights2(i,1)=weights2(i,1)+coeff*delta2*b;
        end
        for i=9:12
            weights2(i,1)=weights2(i,1)+coeff*delta3*c;
        end
        for i=13:16
            weights2(i,1)=weights2(i,1)+coeff*delta4*d;
        end


        weightsout(1,1)=weightsout(1,1)+coeff*deltaout*e;
        weightsout(2,1)=weightsout(2,1)+coeff*deltaout*f;
        weightsout(3,1)=weightsout(3,1)+coeff*deltaout*g;
        weightsout(4,1)=weightsout(4,1)+coeff*deltaout*h;

        achieve(j,1)=reqout(j,1);
        have(j,1)=out;
       % if(loop==200)
        %    have(j,1)=reqout(j,1);
        %end
    end
end

%Testing ANN
desired_out=zeros(size(test(:,1),1),1);
out=zeros(size(test(:,1),1),1);
for i=1:size(test(:,1))
    if(strcmp(test(i),'SKY'))
        desired_out(i,1)=0;
    else
        desired_out(i,1)=1;
    end
end
numIn = size(desired_out,1);
 for j = 1:numIn
        a=perceptron2(weights1(1:19,1),tmp_data.data(j,:));
        b=perceptron2(weights1(1:19,2),tmp_data.data(j,:));
        c=perceptron2(weights1(1:19,3),tmp_data.data(j,:));
        d=perceptron2(weights1(1:19,4),tmp_data.data(j,:));

        e=perceptron2(weights2(1:4,1),[a,b,c,d]);
        f=perceptron2(weights2(5:8,1),[a,b,c,d]);
        g= perceptron2(weights2(9:12,1),[a,b,c,d]);
        h= perceptron2(weights2(13:16,1),[a,b,c,d]);

        out(1,j)=perceptron2(weightsout(1:4,1),[e,f,g,h]);
        if(out(1,j)>0)
            out(1,j)=1;
        else
            out(1,j)=0;
        end
        %out(1,j)=desired_out(j,1);
 end
    
 out2=zeros(2,size(test,1)-1);
  for j=1:size(test,1)
        if(out(1,j)==1)
            out2(1,j)=1;
            out2(2,j)=0;
        else
            out2(1,j)=0;
            out2(2,j)=1;
        end
  end
  
  label2=zeros(2,size(test,1)-1);
    for j=1:size(test,1)
        if(strcmp(test(j,1),'SKY'))
            label2(1,j)=0;
            label2(2,j)=1; 
        else
            label2(1,j)=1;
            label2(2,j)=0;
        end
    end
    
  
  figure;
  plotconfusion(label2,out2)  
  title('Confusion matrix for double hidden layer network with 4 nodes each');

