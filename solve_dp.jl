const m=4  #dimension of the state space
const k=5  #capacity of order quantity
const T=4 #number of periods
const n_sample=10 #sample size to approximate expectation
const drate=4.#demand rate of Poisson process
const h=0.5   #unit holding cost
const r=5.    #unit price
const c=3.    #unit order cost
const theta=2. # unit disposal cost
const s=1.     # unit salvage value
const alpha=0.95  #discount rate
import Distributions

function genDemand(drate)
        demands=zeros(Int,(n_sample, T))
        for t=1:T
            demands[:,t]= rand(Distributions.Poisson(drate),n_sample)
        end
        demands
end



function code(x:: Array{Int,1})
        codex=0
        for i=1:m
            codex += x[i]*(k^(m-i))
        end
        codex
end

function decode(code:: Int)
        x=zeros(Int,m)
        for i=m:(-1):1
            x[m+1-i]= ifloor((code % k^i)/(k^(i-1)))
        end
        x
end


function cost_to_go(x,z,q,D)
        s*z-h* max(sum(x)-z, 0)- alpha *c*q+ alpha *r* min(D, sum(x)+q-z)- alpha *theta* max(x[1]-z-D,0)
end

function dep(x:: Array{Int,1},z)
        x_rem=x[:]
        i=1
        while i<= length(x) && sum(x[1:i])<= z
              i += 1
        end
        i==1 && (x_rem[1]= x[1]- z; return x_rem)
        i== length(x)+1 && return zeros(Int,length(x))
        x_rem[i]= sum(x[1:i])-z
        x_rem[1:(i-1)]=0
        return x_rem
end

function dep_one(x)
        dep(x,1)
end

function dynamic(x,z,q,D)
        x_rem= [x[2:end],q]
        dep(x_rem, max(z+D-x[1],0))
end

function obj(x:: Array{Int,1}, d:: Array{Int,1}, valuevec::Array{Float64,1}, demands)
        function obj_fix(demand)
                y_code= code(dynamic(x,d[1],d[2],demand))+1
                if y_code>= k^m || y_code< 1 println("error",  y_code)
                end
                return alpha* valuevec[y_code ]+ cost_to_go(x,d[1],d[2],demand)
        end
        return mean(map(obj_fix, demands))
end

function optq(y:: Array{Int, 1}, valuevec:: Array{Float64,1}, demands, qmin=0, qmax=k-1)
        lb, ub= qmin, qmax
        lp, rp= iround(lb+ 0.382*(ub-lb)), iround(lb+ 0.618*(ub-lb))
        while ub-lb> 1
              if obj(y,[0,lp], valuevec, demands) <= obj(y, [0,rp], valuevec, demands)
                 lb, ub= lp, ub
                 lp, rp= rp, iround(lb+ 0.618*(ub- lb))
              else
                 lb, ub = lb, rp
                 lp, rp = iround(lb + 0.382* (ub -lb)), lp
              end
        end
        return lb
end

function wrapobj(x:: Array{Int,1}, z::Int, valuevec::Array{Float64, 1}, demands, qmin=0)
        opt_q= optq(dep(x,z), valuevec, demands, qmin)
        return obj(x,[z,opt_q], valuevec, demands)
end

function optz(x, lb, ub, valuevec, demands)
        lp, rp = ifloor(lb + 0.382* (ub -lb)), ifloor(lb + 0.618* (ub - lb))
        while ub- lb > 1
             if wrapobj(x,lp,valuevec, demands) <= wrapobj(x,rp,valuevec, demands)
                lb, ub = lp, ub
                lp, rp = rp, ifloor(lb + 0.618* (ub - lb))
             else
                lb, ub = lb, rp
                lp, rp = ifloor(lb + 0.382* (ub -lb)), lp
             end
        end
        return ub
end

function fillvalue!(state, opt_z, temp, ind, valuevec, demands)
        to_fill_state = decode(state-1)
        index= 0
        while !ind[code(to_fill_state)+1]
              temp[code(to_fill_state)+1]= wrapobj(to_fill_state, max(opt_z- index,0),valuevec,demands)
              ind[code(to_fill_state)+1] = true
#               println(to_fill_state)
              if code(to_fill_state)== 0; break;
              end
              to_fill_state= dep_one(to_fill_state)
              index += 1
        end
end


function mainprog!(valuevec, demands)
        for state= k^m: (-1): (k^m- k+ 1)
           opt_z= optz(decode(state-1), 0, (m)*(k-1), valuevec, demands)
           fillvalue!(state, opt_z, temp, ind, valuevec, demands)
        end
        for i= 1:(m-1)
            for state= (k^m -k^i): (-1): (k^m- k^(i+1)+ 1)
                if ind[state]  continue;
                end
                zstar= optz(decode(state-1), 0, (m-1-i)*(k-1),valuevec, demands)
                if zstar < (m-1-i)*(k-1)
                   opt_z= zstar
                   fillvalue!(state, opt_z, temp, ind, valuevec, demands)
                else
                   @assert ind[state % (k^(i+1))]
                   to_fill_state = decode(state-1)
                   index= 0
                   while !ind[code(to_fill_state)+1] && index<= (m-1-i)*(k-1)
                        temp[code(to_fill_state)+1]= temp[state % (k^(i+1))] + s*((m-1-i)*(k-1)-index)
                        ind[code(to_fill_state)+1] = true
                        to_fill_state= dep_one(to_fill_state)
                        index += 1
                   end
                end
            end
        end
        valuevec[:] = temp[:]
end

#main program

#initialization

Demands= genDemand(drate)
valuevec = zeros(k^m)
for i=1:k^m
    valuevec[i] = s* sum(decode(i-1))
end

temp = zeros(k^m)
ind = fill(false, k^m)

# compute the optimal value for all initial state
init= time()
for t=1:T
    mainprog!(valuevec, Demands[:,t])
    fill!(ind, false)
end

# compute the value function with fluid heuristics
function fluidval_nodep(x:: Array{Int,1}, t)
         t==0 && return s* sum(x)
         @assert  t>= 1
         zf, qf = 0, iround(max(0, drate- sum(x)))
         function helper(demand)
                return alpha* fluidval_nodep(dynamic(x, zf, qf, demand), t-1) + cost_to_go(x, zf, qf,demand)
         end
        return mean(map(helper, Demands[:, t]))
end

function fluidval(x, t= T)
          perish= 0
          for i=1: length(x)
             perish= max(perish, sum(x[1:i])-i*drate)
          end
          zf= iround(perish + max(0, sum(x)-perish- 3* drate))
          return s* zf + fluidval_nodep(dep(x,zf), t)
end


println(valuevec[10:20])
println("The total time elapsed: ", time()-init)

@time fluidval([1,2,3,4])







