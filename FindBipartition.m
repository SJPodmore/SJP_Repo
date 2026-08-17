function [FV] = FindBipartition(A)
% Spectral method for finding bipartite partitions based on the eigenvector
% corresponding to the smallest eigenvalue of the adjacency matrix. 
[V,D] = eig(full(A));
[D,idx] = sort(diag(D));
V=V(:,idx);
V = V(:,1);
FV = zeros(size(A,1),1);
FV(V>=0)=1;
FV(V<0)=2;
end 