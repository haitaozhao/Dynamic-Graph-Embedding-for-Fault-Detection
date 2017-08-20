function [W] = myConstructW(D,K,sign)
%% Input parameters
% D is the input dataset. Each row is a sample.
% K is the number of nearest neighbors.
% sign is 0 or 1. In this paper, sign = 1.

%%  Compute the parameter  $\beta_1$ and $\beta_2$
Data = D';
[row1,col1]=size(Data);
for i=1:col1
   for j=1:col1
       D(i,j)=(Data(:,i)-Data(:,j))'*(Data(:,i)-Data(:,j)); 
       DD(i,j) = (i-j)'*(i-j);
   end   
end

beta = 1/col1^2 * sum(sum(D));
beta = sqrt(beta);
beta2 = 1/col1^2 * sum(sum(DD));
beta2 = sqrt(beta2); 
%% Compute Equation (3)
TempData = zeros(row1,1);
for i = 1 : col1
    for j = i : col1
        TempData = Data(:,i) - Data(:,j);
        Correlation(i,j) = exp(-TempData'*TempData/beta)*exp( -( (j-i)^2)/beta2 );
        Correlation(j,i) = Correlation(i,j);
    end
end

Similarity = zeros(col1);
TempDist = zeros(row1,1);
for i = 1 : col1
    TempDist = D( :,i );
    [~,IndexOfSort] = sort(TempDist);
    for j = 1 : K
        Similarity(i,IndexOfSort(j)) = Correlation(i,IndexOfSort(j));
        Similarity( IndexOfSort(j),i ) = Correlation(i,IndexOfSort(j));
    end
end
S = Similarity;
S(S==0) = 0.01;

%% Compute Equation (4)
for i=1:col1
    SUM_S=sum( S(i,:) );
    for j=1:col1
        P(i,j)=S(i,j)/SUM_S;
    end
end

%% Compute $\alpha$ and A
[Feature_Vector,Eigenvalue]=eig(P');
alpha=Feature_Vector(:,1);
alpha = alpha/sum(alpha);
for i=1:col1
    A(i,:)=alpha';
end
I=eye(col1,col1);

%% Compute Equation (5)
Z=inv(I-P+A);

%% Compute $C_{ij}$
for i=1:col1
    for j=1:col1
        if(i==j)
            phi=1;
        else
            phi=0;
        end
        C(i,j)=alpha(i)*Z(i,j)+alpha(j)*Z(j,i)-alpha(i)*alpha(j)-alpha(i)*phi;
        if C(i,j)<=0
             C(i,j) = 0.0000001;
        end
    end
end

%% Compute Equation (6)
for i=1:col1
    for j=1:col1
        CC(i,j)=C(i,j)/sqrt(C(i,i)*C(j,j)+eps);
    end
end

if sign ==1
    W = CC;
    W = W - diag(ones(1,500));
end
