function [A] = MakeAdjacencyMatrix(Edge)
%Takes an Edge pair list as an input and covert its 
% to an adjacency matrix. Assuming indexing starts at 0
[m,w3]=size(Edge);
n = max(max(Edge(:,1:2)))+1;
if (w3<3)
    Edge = Edge+1; 
    A = sparse(Edge(:,1),Edge(:,2),1,n,n);
else
    Edge(:,1:2) = Edge(:,1:2)+1;
    A = sparse(Edge(:,1),Edge(:,2),Edge(:,3),n,n);
end
%A=full(A);
end