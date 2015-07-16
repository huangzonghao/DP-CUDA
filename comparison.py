
# coding: utf-8


n= 40**5 
f= open("/ghome/hzhangaq/DP-parellel-computing/CCode/dp_optimal_0711.log");
g= open("/ghome/hzhangaq/DP-parellel-computing/CCode2/dp_policy_0711.log");
h= open("/ghome/hzhangaq/DP-parellel-computing/CCode1/dp_spolicy_0711.log");
sum = 0
sum1 = 0
count=0
for i in range(n):
    a1= float(f.readline());
    a2= float(g.readline());
    a3= float(h.readline());
    if((a1-a3)/a1 < 0.05):
      count += 1;
    sum += (a1- a2)/a1;
    sum1 += (a1-a3)/a1;

sum= sum/n  * 100
sum1= sum1/n * 100  
print sum, sum1





