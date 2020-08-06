function [W,b,alpha,gndu,obj] = JCD(X,gnd,Parameter)
% X: cells£»X{v}: num\times dim(v).
% gnd: labels of firsr l data points.
% Parameter: p,r,lam. p can be fixed as 0.5. r is varied in the range of {1.1,1.3, 1.5,1.7,1.9,2.5,3.3}. lam is varied in the range of {1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1}.
% W: cells; b:cells; 
% alpha: the view weight vector.
% gndu: labels of unlabeled training data points.
p = Parameter.p;
r = Parameter.r;
lam = Parameter.lam;
maxIter = 30;
thresh = 1e-4;
View = max(size(X));
n = size(X{1},1);
l = max(size(gnd));
C = max(gnd);

Yl = -1*ones(l,C);
for i = 1:l
    Yl(i,gnd(i))=1;
end
ET = -1*ones((n-l),C);
%% Initilization
for v = 1:View
    Ml{v} = zeros(l,C);
    viewDim(v) = size(X{v},2);
    for k = 1:C
        Mu{v}{k} = zeros((n-l),C);
    end
    alpha(v)=1/View;
    Ix = [X{v},ones(n,1)];
    if viewDim(v) < l
       G = Ix(1:l,:)'*Ix(1:l,:)+lam*speye(viewDim(v)+1);
       Wb = G\Ix(1:l,:)'*Yl;
    else
       G = Ix(1:l,:)*Ix(1:l,:)'+lam*speye(l);
       Wb= Ix(1:l,:)'*(G\Yl);
    end
    W{v} = Wb(1:(end-1),:);
    b{v} = Wb(end,:);
end

for iter=1:maxIter
fprintf('newMSS iteration %d...\n', iter);
%% update Y
if iter==1
Yut = zeros((n-l),C);
for v = 1:View
    TT = X{v}*W{v}+ones(n,1)*b{v};
    Tl{v} = TT(1:l,:);
    Tu{v} = TT((l+1):n,:);
    Mp{v} = zeros((n-l),C);
    YP = (Tu{v}-Mp{v}.*ET-ET).^2;
    YS = sum(YP,2);
    for k = 1:C
       YP2 = Tu{v}(:,k)-Mu{v}{k}(:,k).*ones((n-l),1)-ones((n-l),1); 
       HH{v}(:,k)=(YS-YP(:,k)+YP2.^2);
       Yut(:,k)=Yut(:,k)+alpha(v)*HH{v}(:,k);
    end
end
end
if (r==1)
    Yu = zeros((n-l),C);
    [~,YII]=min(Yut');
    for i = 1:(n-l)
        Yu(i,YII(i)) = 1;
    end
else
    Yut = (r.*Yut).^(1/(1-r));
    s0=sum(Yut,2);%length(find(isnan(s0)))
    Yu=Yut./repmat(s0,1,C);
end
YuF = Yu.^(r);

%% update M(v)
for v = 1:View
    Ml{v} = max(Yl.*(Tl{v}-Yl),0);
    Mp{v} = max(ET.*(Tu{v}-ET),0);
    for k = 1:C
        Mu{v}{k} = Mp{v};
        mtk{v}(:,k)=max(ones((n-l),1).*(Tu{v}(:,k)-ones((n-l),1)),0);
        Mu{v}{k}(:,k) = mtk{v}(:,k);
    end
end
%% update W(v) 
uu = [ones(l,1);sum(Yu.^(r),2)];
zz = sum(uu);
for v=1:View
    TM = sum(YuF.*Mp{v},2);
    TFl = Ml{v}.*Yl+Yl;
    TFu = zeros(n-l,C);
    for k=1:C
      TFu(:,k) = 2*YuF(:,k)-uu((l+1):end)+YuF(:,k).*mtk{v}(:,k)-TM+YuF(:,k).*Mp{v}(:,k);
    end
    F = [TFl;TFu];
    Fx = F-repmat(uu,1,1).*ones(n,1)*b{v};
    if viewDim(v) < n
       G = X{v}'*(repmat(uu,1,viewDim(v)).*X{v})+lam*speye(viewDim(v));
       W{v}=G\(X{v}'*Fx);
    else
       if iter == 1
           XXT{v} = X{v}*X{v}';
       end
       G = XXT{v};
       G(1:size(G,1)+1:end) = G(1:size(G,1)+1:end)+lam*(uu.^(-1))';
       W{v}= X{v}'*(G\(repmat((uu.^(-1)),1,C).*Fx));
    end
    T{v} = X{v}*W{v};
    b{v}=ones(1,n)*(F-repmat(uu,1,C).*T{v})/zz;
    TT = T{v}+ones(n,1)*b{v};
    Tl{v} = TT(1:l,:);
    Tu{v} = TT((l+1):n,:);
end
%% update alpha
Yut = zeros((n-l),C);
for v = 1:View
    YP = (Tu{v}-Mp{v}.*ET-ET).^2;
    YS = sum(YP,2);
    for k = 1:C
       YP2 = Tu{v}(:,k)-Mu{v}{k}(:,k).*ones((n-l),1)-ones((n-l),1); 
       HH{v}(:,k)=(YS-YP(:,k)+YP2.^2);
       Yut(:,k)=Yut(:,k)+alpha(v)*HH{v}(:,k);
    end
end
alpha = zeros(View,1);
for v=1:View
objview(v) = norm(Tl{v}-Ml{v}.*Yl-Yl,'fro')^2+sum(sum(YuF.*HH{v}))+lam*sum(sum(W{v}.^2));
alpha(v)=abs(objview(v).^(p-1));
objview(v) = objview(v).^p;
end
obj(iter) = (sum(objview)/View)^(1/p);  
if(iter > 2)
    obj_diff = (obj(iter-1) - obj(iter))/obj(iter-1);
    if(obj_diff < thresh)
        break;
    end
end
end
[~,gndu]=max(Yu');