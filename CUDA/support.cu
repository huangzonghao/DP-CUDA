#include "support.h"
#include "parameters.h"
//the MSB is the number of items to be expired and the LSB is the number of the newly purchased items
/******** kernels ********/

/* power function */
__device__ 
long ipow(size_t base, size_t exp)
{
    long result = 1;
    while (exp != 0)
    {
        if ((exp & 1) == 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }

    return result;
}
/* convert the 1d index to m-d coordinate */
__device__
void 1DtomD(long 1DIdx, size_t* mDIdx, size_t k, size_t m){

      for( size_t i = 0; i < m ; ++i){
            mDIdx[i] = 1DIdx % k;
            1DIdx /= k;
      }
      return;

}

/* convert the mD coordinate to 1d index */
__device__ 
void mDto1D(size_t* mDIdx, long &1DIdx, size_t k, size_t m){
        long result = 0;
        for (size_t i = 0; i < m; ++i){
               result += mDIdx[i] * ipow(k, i);     // can be optimized once set up a reference table for ipow(k,i)
        }

        1DIdx = result;
}

/* returns the total number of items stored */
__device__
size_t checkStorage(size_t* mDarray, size_t m){
      size_t result = 0;
     for (size_t i = 0; i < m ; ++i ){
             result += mDarray[i];
     }
     return result;
}
/* depleting */
/* note the MSB represents the number of items to be expired  */
/* so pay attention to the array index
 */
__device__
void depleteStorage( size_t* mDarray,  size_t d_amount){
        size_t buffer = 0;
        size_t i = 0;
        while(!d_amount && i < m){ 
                if ( !mDarray[i]){
                        ++i;
                        continue;
                }

                if(mDarray[i] - d_amount >= 0)
                {
                        mDarray[i] -= d_amount;
                        d_amount  = 0;
                        break;
                }
                buffer = d_amount - mDarray[i];
                mDarray[i] = 0;
                d_amount = buffer;
                buffer = 0;
                ++i;
        }
}

/* set the init value of all entries in the value table to 0 */
__global__ 
void valueTableInit(float* d_valueTable, long arrayLength  ){    
  long stepSize = gridDim.x * gridDim.y * blockDim.x * blockDim.y;  // the total number of threads which have been assigned for this task
  long myStartIdx = (gridDim.x * blockIdx.y + blockIdx.x - 1) * blockDim.x * blockDim.y +  threadIdx.y * blockDim.x + threadIdx.x;
  for (long long i = myStartIdx; i < arrayLength; i += stepSize)
    d_valueTable[i] = 0;

  __syncthreads(); 

}

/* evaluate the state value given z and q */
/* return th expected value over the demands */
__device__
float stateValue( size_t dataIdx, size_t expiringToday, size_t storageToday, size_t z, size_t q, size_t * s_demandWeight, size_t numDemands){
         float profit = 0;
         float sum = 0;
         for ( size_t i = 0; i < numDemands; ++i){
                 profit = s * z                                                             // the money collected depletion 
                        - h * max(int(storageToday) - z , 0)                                // the cost for holding all the items 
                        - alpha * c * q                                                     // the money spent on ordering new items
                        + alpha * r * min(i, storageToday - z + q)               // the total income from selling the products to the customers
                        - alpha * theta * max(expiringToday - z - i, 0);// the money spent on the expired items
                 sum += profit * s_demandWeight[i];
         }

         return sum;
}


/* Initialize the cuda device */
/* 1. allocate the device memory and initialize the values (the data type is hard coded to float)
 * 2. copy the random table to device
 */
void deviceInit(float * h_demandWeights, float * d_demandWeights, float * d_valueTable, float * d_tempTable, int d_zTable, size_t demandTableSize, long long valueTableSize){
       // memory allocation 
        checkCudaErrors(cudaMalloc( &d_demandWeights, demandTableSize * sizeof(float)));
        checkCudaErrors(cudaMallor( &d_valueTable, valueTableSize * sizeof(float)));
        checkCudaErrors(cudaMallor( &d_tempTable, valueTableSize * sizeof(float)));
        checkCudaErrors(cudaMallor( &d_zTable, valueTableSize * sizeof(int)));
        
       // copy data
        checkCudaErrors(cudaMemcpy( d_demandWeights, h_demandWeights, demandTableSize * sizeof(float)));

       // init the value table
       // we may use as many as threads possible here

       dim3 gridSize(1600, 1, 1);
       dim3 blockSize(1024, 1, 1);
       valueTableInit<<<gridSize, blockSize>>>(d_valueTable, valueTableSize);
       // init the temp table
       valueTableInit<<<gridSize, blockSize>>>(d_tempTable, valueTableSize);
       // init the z value table
       valueTableInit<<<gridSize, blockSize>>>(d_zTable, valueTableSize);

       return;
}

