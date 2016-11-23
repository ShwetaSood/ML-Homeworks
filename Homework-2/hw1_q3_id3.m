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

%Discretization of Data
class=['GRASS','BRICKFACE','PATH','WINDOW','CEMENT','FOLIAGE','SKY'];
out=zeros(size(tmp_data.data));
for j=1:size(tmp_data.data,2)
    thres=mean(tmp_data.data(:,j));
    for i=1:size(tmp_data.data,1)
        if (tmp_data.data(i,j)<=thres)
            out(i,j)=0;
        else
            out(i,j)=1;
        end
        
        if strcmp(training(i+1,1),'GRASS')
            out(i,20)=1;
        elseif strcmp(training(i+1,1),'BRICKFACE')
            out(i,20)=2;   
        elseif strcmp(training(i+1,1),'PATH')
            out(i,20)=3;
        elseif strcmp(training(i+1,1),'WINDOW')
            out(i,20)=4;
        elseif strcmp(training(i+1,1),'CEMENT')
            out(i,20)=5;
        elseif strcmp(training(i+1,1),'FOLIAGE')
            out(i,20)=6;
        elseif strcmp(training(i+1,1),'SKY')
            out(i,20)=7;
        end
        
    end
end

testout=zeros(size(testtmp_data.data));
for j=1:size(testtmp_data.data,2)
    thres=mean(testtmp_data.data(:,j));
    for i=1:size(testtmp_data.data,1)
        if (testtmp_data.data(i,j)<=thres)
            testout(i,j)=0;
        else
            testout(i,j)=1;
        end
        if strcmp(test(i+1,1),'GRASS')
            testout(i,20)=1;
        elseif strcmp(test(i+1,1),'BRICKFACE')
            testout(i,20)=2;   
        elseif strcmp(test(i+1,1),'PATH')
            testout(i,20)=3;
        elseif strcmp(test(i+1,1),'WINDOW')
            testout(i,20)=4;
        elseif strcmp(test(i+1,1),'CEMENT')
            testout(i,20)=5;
        elseif strcmp(test(i+1,1),'FOLIAGE')
            testout(i,20)=6;
        elseif strcmp(test(i+1,1),'SKY')
            testout(i,20)=7;
        end
        
    end
end

list_att=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19];
final_list=zeros(1,2000);

index=1; %for recursive changed to 1
%Recursive
data=out;
datalist=cell(1,100);
datalist(1,1)={data};
iter=2;
for index=1:53
   if(datalist{1,index}~=-1)
        [list_att,final_list,data1,data2]=call_tree(list_att,final_list,datalist{1,index},index);
        datalist{1,iter}=data1;
        datalist{1,iter+1}=data2;
        iter=iter+2;
   else
       final_list(index)=0;
   end
end

%{
index=0;
%Iteration 1
att_no=attribute_no_choose(list_att,out);
list_att(att_no)=-1;
index=index+1;
final_list(index)=att_no;
[data1,data2]=datasets(out,att_no);

%Iteration 2
att_no=attribute_no_choose(list_att,data1);
if (att_no~=-1)
    list_att(att_no)=-1;
    index=index+1;
    final_list(index)=att_no;
    [data3,data4]=datasets(data1,att_no);
else
    index=index+2;
end

att_no=attribute_no_choose(list_att,data2);
if (att_no~=-1)
    list_att(att_no)=-1;
    index=index+1;
    final_list(index)=att_no;
    [data5,data6]=datasets(data2,att_no);
else
    index=index+2;
end

%Iteration 3
att_no=attribute_no_choose(list_att,data3);
if (att_no~=-1)
    list_att(att_no)=-1;
    index=index+1;
    final_list(index)=att_no;
    [data7,data8]=datasets(data3,att_no);
else
    index=index+2;
end

att_no=attribute_no_choose(list_att,data4);
if (att_no~=-1)
    list_att(att_no)=-1;
    index=index+1;
    final_list(index)=att_no;
    [data9,data10]=datasets(data4,att_no);
else
    index=index+2;
end

att_no=attribute_no_choose(list_att,data5);
if (att_no~=-1)
    list_att(att_no)=-1;
    index=index+1;
    final_list(index)=att_no;
    [data11,data12]=datasets(data5,att_no);
else
    index=index+2;
end

att_no=attribute_no_choose(list_att,data6);
if (att_no~=-1)
    list_att(att_no)=-1;
    index=index+1;
    final_list(index)=att_no;
    [data13,data14]=datasets(data6,att_no);
else
    index=index+2;
end
%}
%Accuracy testing
achieved=zeros(size(testout,1),1);
desired=testout(:,20);
for i=1:size(testout,1)
    j=1;
    while(j<length(final_list) && final_list(1,j)~=0)
        att=final_list(1,j);
        if(testout(i,att)==0)
            j=2*j;
        else
            j=2*j+1;
        end
    end
    if(datalist{1,j}~=-1)
        ele=datalist{1,j};
        achieved(i,1)=mode(ele(:,20));
    end
end
cnt=0;
for i=1:size(achieved,1)
    if(achieved(i,1)==desired(i,1))
        cnt=cnt+1;
    end
end
a=double(cnt/210)*100;
b=double(cnt/210)*100;
disp(['Absolute Classification Accuracy of the model is : ' num2str(a)]);
disp(['Mean Class-Wise Accuracy of the model is : ' num2str(b)]);