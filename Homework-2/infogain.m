function [ infogain ] = infogain( data,attribute_no)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
Ent=entropy(data);
size1=length(find(data(:,attribute_no)==0));
size2=size(data,1)-size1;
totalsize=size1+size2;
cols=size(data,2);
data1=zeros(size1,cols); %Contain 0 values for attributes
data2=zeros(size2,cols); %Contain 1 values for attributes
k1=1;
k2=1;
for i=1:size(data,1)
       if( data(i,attribute_no)==0)
           data1(k1,:)=data(i,:);
           k1=k1+1;
       else
           data2(k2,:)=data(i,:);
           k2=k2+1;
       end
end
infogain= Ent - (size1/totalsize * entropy(data1) + size2/totalsize * entropy(data2));
end

