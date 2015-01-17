const m=4  #dimension of the state space
const k=10 #capacity of order quantity
const T=5 #number of periods
const n_sample=10#sample size to approximate expectation
const drate=8.#demand rate of Poisson process
const h=0.5   #unit holding cost
const r=5.    #unit price
const c=3.    #unit order cost
const theta=2. # unit disposal cost
const s=1.     # unit salvage value
const alpha=0.95  #discount rate
const maxhold= 3
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
            x[m+1-i]= ifloor((code - ifloor(code/ k^i) * k^i)/(k^(i-1)))
        end
        x
end

function cost_to_go(x,z,q,D)
        return s*z-h* max(sum(x)-z, 0)- alpha *c*q+ alpha *r* min(D, sum(x)+q-z)- alpha *theta* max(x[1]-z-D,0)
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
        dep([x[2:end],q], max(z+D-x[1],0))
end

function obj(x:: Array{Int,1}, d:: Array{Int,1}, valuevec::Array{Float64,1}, demands)
        totalsum= 0.
        for demand in demands
                totalsum += alpha* valuevec[code(dynamic(x,d[1],d[2],demand))+1]+ cost_to_go(x,d[1],d[2],demand)
        end
        return totalsum/n_sample
end

function optq(y:: Array{Int, 1}, valuevec:: Array{Float64,1}, demands, qmin=0, qmax=k-1)
        lb, ub= qmin, qmax
        lp, rp= ifloor(lb+ 0.382*(ub-lb)), ifloor(lb+ 0.618*(ub-lb))
        while ub-lb> 1
              if obj(y,[0,lp], valuevec, demands) <= obj(y, [0,rp], valuevec, demands)
                 lb, ub= lp, ub
                 lp, rp= rp, ifloor(lb+ 0.618*(ub- lb))
              else
                 lb, ub = lb, rp
                 lp, rp = ifloor(lb + 0.382* (ub -lb)), lp
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
        currentvalue = wrapobj(decode(state -1), opt_z, valuevec,demands)
        to_fill_state = decode(state-1)
        index= 0
        while !ind[code(to_fill_state)+1]
              if index <= opt_z
                 temp[code(to_fill_state)+1]= currentvalue- s*index
              else
                 temp[code(to_fill_state)+1] = wrapobj(to_fill_state, 0,valuevec,demands)
              end
              ind[code(to_fill_state)+1] = true
#               println(to_fill_state)
              if code(to_fill_state)== 0; break;
              end
              to_fill_state= dep_one(to_fill_state)
              index += 1
        end
        nothing
end

function premainprog!(temp, ind, valuevec, demands)
        for state= k*(k-1)+ (k-1) :(-1): k*(k-1)
           opt_z= optz(decode(state-1), 0, sum(decode(state-1)), valuevec, demands)
           fillvalue!(state, opt_z, temp, ind, valuevec, demands)
        end
        nothing
end

function mainprog!(temp, ind, jobno, valuevec, demands)
        primestate= decode(k^m-k+jobno-1)
        opt_z= optz(primestate,0, sum(primestate), valuevec, demands)
        fillvalue!(k^m-k+jobno, opt_z, temp, ind, valuevec, demands)
        for i= 0:(m-2)
            for index= (k^(m-1) -k^i): (-1): (k^(m-1)- k^(i+1)+ 1)
                state= (index-1)* k+ jobno
                if ind[state]  continue;
                end
                zstar= optz(decode(state-1), 0, (m-2-i)*(k-1),valuevec, demands)
                @assert ind[state % (k^(i+2))]
                if zstar < (m-2-i)*(k-1)
                   opt_z= zstar
                   fillvalue!(state, opt_z, temp, ind, valuevec, demands)
                else
                   to_fill_state = decode(state-1)
                   index= 0
                   while !ind[code(to_fill_state)+1] && index <= (m-2-i)*(k-1)
                        temp[code(to_fill_state)+1]= temp[state % (k^(i+2))] + s*((m-2-i)*(k-1)-index)
                        ind[code(to_fill_state)+1] = true
                        to_fill_state= dep_one(to_fill_state)
                        index += 1
                   end
                end
            end
        end
        nothing
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

# First Part: compute the optimal value for all initial states
starttime= time()
for t=1:T
    premainprog!(temp, ind, valuevec, Demands[:,t])
    for jobno= 1:k
      mainprog!(temp, ind, jobno, valuevec, Demands[:,t])
    end
    @assert all(ind)
    valuevec[:]= temp[:]
    fill!(ind, false)
end

# Second Part: compute the value function with fluid heuristics
function fluidval_zerodep(x:: Array{Int,1}, t)
         t==0 && return s* sum(x)
         @assert  t>= 1
         zf, qf = 0, iround(max(0, drate- sum(x)))
         value=0.
         for demand in Demands[:,t]
                value += alpha* fluidval_zerodep(dynamic(x, zf, qf, demand), t-1) + cost_to_go(x, zf, qf,demand)
         end
         return value/n_sample
end

function fluidval(x, t= T)
          perish= 0
          for i=1: length(x)
             perish= max(perish, sum(x[1:i])-i*drate)
          end
          fluid_z= iround(perish + max(0, sum(x)-perish- maxhold* drate))
          return s* fluid_z + fluidval_zerodep(dep(x,fluid_z), t)
end



# print some outputs of the main program.
println(valuevec[10:20]," ", fluidval([1,2,3,4]))
println("The total time elapsed: ", time()-starttime)







