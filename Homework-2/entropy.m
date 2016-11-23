function [ Ent ] = entropy( data )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
Ent=0;
total=size(data(:,20),1);
len=length(unique(data(:,20)));
probability=zeros(1,len);
values=unique(data(:,20)); %all unique classes
for i=1:len
    cnt=length(find(data(:,20)==values(i)));
    probability(1,i)=double(cnt/total);
end

for i=1:len
    Ent=Ent+probability(1,i)*log2(probability(1,i))*-1;
end
end

