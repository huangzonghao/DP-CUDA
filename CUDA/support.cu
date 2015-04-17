#include "support.h"
#include "parameters.h"

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>

//the MSB is the number of items to be expired and the LSB is the number of the newly purchased items
/******** kernels ********/


/* power function */
__device__ inline
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
__device__ inline
void 1DtomD(long 1DIdx, size_t* mDIdx){

      for( size_t i = 0; i < m ; ++i){
            mDIdx[i] = 1DIdx % k;
            1DIdx /= k;
      }
      return;

}

/* convert the mD coordinate to 1d index */
__device__ inline
void mDto1D(size_t* mDIdx, long &1DIdx){
        long result = 0;
        for (size_t i = 0; i < m; ++i){
               result += mDIdx[i] * ipow(k, i);     // can be optimized once set up a reference table for ipow(k,i)
        }

        1DIdx = result;
}

/* returns the total number of items stored */
__device__ inline
size_t checkStorage(size_t* mDarray){
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
__device__ inline
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
void valueTableSettoZero(float* d_valueTable, long arrayLength  ){    
  long stepSize = gridDim.x * gridDim.y * blockDim.x * blockDim.y;  // the total number of threads which have been assigned for this task
  long myStartIdx = (gridDim.x * blockIdx.y + blockIdx.x - 1) * blockDim.x * blockDim.y +  threadIdx.y * blockDim.x + threadIdx.x;
  for (long long i = myStartIdx; i < arrayLength; i += stepSize)
    d_valueTable[i] = 0;

  __syncthreads(); 

}

/* evaluate the state value given z and q */
/* return th expected value over the demands */
__device__ inline
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

/* use one d arrangement here */
__global__ 
void valueTableUpdateKernel( float* d_randomTable,
                             float* d_valueTable, 
                             float* d_tempTable,
                             size_t iteratingDim,       // the coordinate that we are processing now
                             size_t batchIdx           // the batch index of the kernel launch for that coordinate... together with the iteratingDim, we should be able to obtain the data index
                             ){

  // allocate the shared memory for the demands
  extern __shared__ float s_demands[];

  // the cascading information is given by the iteratingDim
  // assume all the nodes here has the oversight -- the starting several elements are computed by the cpu
  int myIdx = blockIdx.x * blockDim + threadIdx.x;
  long dataIdx = iteratingDim * ipow()



}

/* Initialize the cuda device */
/* allocate the device memory and initialize the values (the data type is hard coded to float)*/
void deviceTableInit(size_t numTables, float ** tables, long * tableLengths, cudaInfoStruct * cudainfo){
       dim3 gridSize(cudainfo->numBlocks, 1, 1);
       dim3 blockSize(cudainfo->numThreadsPerBlock, 1, 1);

       for ( size_t i = 0; i < numTables; ++i){
               checkCudaErrors(cudaMalloc(tables[i], tableLengths[i]));
               valueTableSettoZero<<< gridSize, blockSize>>>(tables[i], tableLengths[i]);
       } 

       return;
} 

/* the data transmission between host and device, the host addr always come first */
void passToDevice(float* h_array, float* d_array, size_t length){
        checkCudaErrors(cudaMemcpy(d_array, h_array, length, cudaMemcpyHostToDevice));
        return;
}
void readFromDevice(float * h_array, float* d_array, size_t length){
        checkCudaErrors(cudaMemcpy(h_array, d_array, length, cudaMemcpyDeviceToHost));
        return;
}


/* update the value table for one day */
/* only need to hold 2 tables and update each one at a time */
void valueTableUpdate( float* d_valueTable, 
                       const dim3 kernelSize,       // number of blocks launched 
                       const dim3 blockSize ){      // number of threads per block launched

}
/* write in the values of the last day in the period */
/**/
__global__
void writeinEdgeValues(float * d_valueTable){

}

/* the interface for the main function */
void presetValueTable(float * d_valueTable, cudaInfoStruct * cudainfo){
  writeinEdgeValues<<<cudainfo->numBlocks, cudainfo->numThreadsPerBlock>>>(float * d_valueTable);
  return;
}
/* the main function which takes care of the dispatch and deploy */
void evalWithPolicy(float* h_valueTable, float * d_valueTables, cudaInfoStruct * cudainfo){
        // first init make the d_valueTables[0] to the edge values
        /* T periods in total */
        for ( size_t iPeriod = 0; iPeriod < T; ++iPeriod){
            // determine which the role of the tables
            valueTableUpdate(

        }
}

/* Gather the system information, fol auto fill in the block number and the thread number per block */
void gatherSystemInfo(size_t * deviceCount, size_t * numBlocks, size_t * numThreadsPerBlock){
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  cudaGetDeviceCount(deviceCount);

  *numThreadsPerBlock = deviceProp.maxThreadsPerBlock;
  numBlocks = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
  return;

}
