
# coding: utf-8


n= 10**7
f= open(“/ghome/hzhangaq/DP-parellel-computing/CCode/dp_optimal_2.log”);
g= open(“/ghome/hzhangaq/DP-parellel-computing/CCode2/dp_policy_2.log”);
sum = 0
for i in range(n):
    a1= float(f.readline());
    a2= float(g.readline());
    sum+=  a1- a2;

sum= sum/n
print sum





