function [ list_att,final_list,data1,data2] = call_tree( list_att,final_list,data,index)
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here
att_no=attribute_no_choose(list_att,data);
if (att_no~=-1 && size(data,1)>20 )
    %disp(['Enter ' num2str(index) '  ' num2str(att_no) ]);
    list_att(att_no)=-1;
    %index=index+1;
    final_list(index)=att_no;
    [data1,data2]=datasets(data,att_no);
   % call_tree(list_att,final_list,data1,2*index);
   % call_tree(list_att,final_list,data2,2*index+1);
else
     final_list(index)=0;
    data1=-1;
    data2=-1;

end


end

