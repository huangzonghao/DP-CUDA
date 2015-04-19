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
/* convert the oneD index to m-d coordinate */
__device__ inline
void oneDtomD (long oneDIdx, size_t* mDIdx){

      for( size_t i = 0; i < m ; ++i){
            mDIdx[i] = oneDIdx % k;
            oneDIdx /= k;
      }

}

/* convert the mD coordinate to oneD index */
__device__ inline
void mDtooneD(size_t* mDIdx, long &oneDIdx){
        long result = 0;
        for (size_t i = 0; i < m; ++i){
               result += mDIdx[i] * ipow(k, i);     // can be optimized once set up a reference table for ipow(k,i)
        }

        oneDIdx = result;
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

/* the data transmission between host and device, the host addr always come first */
void passToDevice(float* h_array, float* d_array, size_t length){
        checkCudaErrors(cudaMemcpy(d_array, h_array, length, cudaMemcpyHostToDevice));
        return;
}
void readFromDevice(float * h_array, float* d_array, size_t length){
        checkCudaErrors(cudaMemcpy(h_array, d_array, length, cudaMemcpyDeviceToHost));
        return;
}

/* Initialize the cuda device */
/* set the init value of all entries in the value table to 0 */
__global__ 
void kernel_deviceTableInit(float* d_valueTable, long arrayLength  ){    
  long stepSize = gridDim.x * gridDim.y * blockDim.x * blockDim.y;  // the total number of threads which have been assigned for this task
  long myStartIdx = (gridDim.x * blockIdx.y + blockIdx.x - 1) * blockDim.x * blockDim.y +  threadIdx.y * blockDim.x + threadIdx.x;
  for (long long i = myStartIdx; i < arrayLength; i += stepSize)
    d_valueTable[i] = 0;

  __syncthreads(); 

}
/* allocate the device memory and initialize the values (the data type is hard coded to float)*/
void deviceTableInit(size_t numTables, float ** tables, unsigned long tableLengths, cudaInfoStruct * cudainfo){
       dim3 gridSize(cudainfo->numBlocks, 1, 1);
       dim3 blockSize(cudainfo->numThreadsPerBlock, 1, 1);

       for ( size_t i = 0; i < numTables; ++i){
               checkCudaErrors(cudaMalloc(tables[i], tableLengths));
               deviceTableInitKernel<<< gridSize, blockSize>>>(tables[i], tableLengths[i]);
       } 

       return;
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
void kernel_valueTableUpdateWithPolicy( float* d_randomTable,
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
/* update the value table for one day */
/* only need to hold 2 tables and update each one at a time */
void valueTableUpdateWithPolicy( float** d_valueTables, 
                                 size_t currentTable, 
                                 size_t numDepletion, 
                                 cudaInfoStruct cudainfo ){

}


/* write in the values of the last day in the period */

__global__
void kernel_presetValueTable(float * d_valueTable, long long table_length){
  extern __shared__ size_t mDIdx[];
  long stepSize = gridDim.x * blockDim.x;  // the total number of threads which have been assigned for this task, oneD layout everywhere
  long myStartIdx = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned long i = myStartIdx; i < table_length; i += stepSize){
    oneDtomD(i,mDIdx);
    d_valueTable[i] = checkStorage(mDIdx) * s;
  }

  __syncthreads(); 
}

/* the interface for the main function */
void presetValueTable(float * d_valueTable, unsigned long  table_length, cudaInfoStruct * cudainfo){
  kernel_presetValueTable<<<cudainfo->numBlocks, cudainfo->numThreadsPerBlock, m * sizeof(size_t) >>>(float * d_valueTable, size_t table_length);
  return;
}
/* the main function which takes care of the dispatch and deploy */
void evalWithPolicy(float* h_valueTable, float ** d_valueTables, unsigned long tableLength, cudaInfoStruct * cudainfo){
// the policy is : the depletion amount is always zero except the first day
        size_t currentTable = 0;      // the index of table to be updated in next action

        /*first init make one of the d_valueTables to the edge values*/
        presetValueTable(d_valueTables[currentTable], valueTablesLength, cudainfo);
        currentTable = 1 - currentTable;

        /* T periods in total */
        for ( size_t iPeriod = 0; iPeriod < T; ++iPeriod){
          if(iPeriod != T-1){
            valueTableUpdateWithPolicy( d_valueTables, currentTable, 0, cudainfo);
            currentTable = 1 - currentTable;
          }
        }
        // the final result stores in the (1 - currentTable)
        readFromDevice(h_valueTable, d_valueTables[1 - currentTable], tableLength);
        return;
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
