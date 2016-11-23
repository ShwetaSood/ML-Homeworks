function [ att_no ] = attribute_no_choose( list_att,out )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
max=-inf;
for i=1:length(list_att)
    if(list_att(i)~=-1)
        if(infogain(out,i)>max)
            max=infogain(out,i);
            att_no=i;
            if max==0
                att_no=-1;
            end
        end
    end
end

end

