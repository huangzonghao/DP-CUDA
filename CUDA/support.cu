#include "support.h"
#include "parameters.h"
#include <iostream>


// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <iostream>
#include "helper_cuda.h"
#include "helper_math.h"
using namespace std;


extern size_t  valueTablesLength;

//the MSB is the number of items to be expired and the LSB is the number of the newly purchased items
/******** kernels ********/

/* power function */
__device__ inline
size_t ipow(size_t base, size_t exp)
{
    size_t result = 1;
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
/* take mD as the base */
__device__ inline
void decode (size_t* mDIdx, size_t oneDIdx){

      for( size_t i = 0; i < m ; ++i){
            mDIdx[m - 1 - i] = oneDIdx % k;
            oneDIdx /= k;
      }

}

/* convert the mD coordinate to oneD index */
__device__ inline
void encode(size_t* mDIdx, size_t* oneDIdx){
        size_t result = 0;
        for (size_t i = 0; i < m; ++i){
               result += mDIdx[i] * ipow(k, m - 1 - i);     // can be optimized once set up a reference table for ipow(k,i)
        }

        *oneDIdx = result;
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

                if(mDarray[i] >= d_amount )
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


// if you don't know how to do perfectly overloading, don't use the following shortcuts....
/* the data transmission between host and device, the host addr always come first */
void passToDevice(float* h_array, float* d_array, size_t length){
        checkCudaErrors(cudaMemcpy(d_array, h_array, length * sizeof(float), cudaMemcpyHostToDevice));
        return;
}

/* since we are only deal with floating points in gpu, we may hard coded the data type to be float */
void passToDevice(const float* h_array, float* d_array, size_t length){
        checkCudaErrors(cudaMemcpy(d_array, h_array, length * sizeof(float), cudaMemcpyHostToDevice));
        return;
}
void readFromDevice(float * h_array, float* d_array, size_t length){
        checkCudaErrors(cudaMemcpy(h_array, d_array, length * sizeof(float), cudaMemcpyDeviceToHost));
        return;
}


/* Initialize the cuda device */
/* set the init value of all entries in the value table to 0 */
__global__ 
void kernel_deviceTableInit(float* d_valueTable, size_t arrayLength  ){    
  size_t stepSize = gridDim.x * gridDim.y * blockDim.x * blockDim.y;  // the total number of threads which have been assigned for this task
  size_t myStartIdx = (gridDim.x * blockIdx.y + blockIdx.x - 1) * blockDim.x * blockDim.y +  threadIdx.y * blockDim.x + threadIdx.x;
  for (size_t i = myStartIdx; i < arrayLength; i += stepSize)
    d_valueTable[i] = 0;

  __syncthreads(); 

}
/* allocate the device memory and initialize the values (the data type is hard coded to float) */
void deviceTableInit(size_t numTables, float ** tables, size_t tableLengths, cudaInfoStruct * cudainfo){
       dim3 gridSize(cudainfo->numBlocks, 1, 1);
       dim3 blockSize(cudainfo->numThreadsPerBlock, 1, 1);

       for ( size_t i = 0; i < numTables; ++i){
               checkCudaErrors(cudaMalloc(&tables[i], tableLengths * sizeof(size_t)));
               kernel_deviceTableInit<<< gridSize, blockSize>>>(tables[i], tableLengths);
       } 

       return;
}



/* evaluate the state value given z and q */
/* return th expected value over the demands */
// to complicit to be an inline function
__device__
float stateValue( size_t expiringToday, 
                  int storageToday, 
                  int z, int q, 
                  float * d_randomTable){

        float profit = 0;
        float sum = 0;
        for ( size_t i = min_demand; i < max_demand; ++i){
                profit = s * z                                                             // the money collected depletion 
                       - h * max(int(int(storageToday) - z) , 0)                                // the cost for holding all the items 
                       - alpha * c * q                                                       // the money spent on ordering new items
                       + alpha * r * min(int(i), int(storageToday - z + q))               // the total income from selling the products to the customers
                       - alpha * theta * max(int(expiringToday - z - i), 0);// the money spent on the expired items
                
                sum += profit * d_randomTable[i];
        }

        return sum;
}

/* use one d arrangement here */
__global__ 
void kernel_valueTableUpdateWithPolicy(  float* d_randomTable,
                                         float* d_valueTable,     // note both the value table and the temp table here hold the exact starting index for this kernel launch
                                         float* d_tempTable,
                                         size_t depletionIndicator,
                                         size_t batchIdx
                                         ){
  // declare the local variables 
  extern __shared__ size_t mDIdx[];
  float bestresult = 0;
  //float bestq = 0;
  float tempresult = 0;
  size_t storageToday = 0;
  // this is both the thread index and the data index in this batch
  size_t myIdx = blockIdx.x * gridDim.x + threadIdx.x;
  size_t dataIdx = myIdx + batchIdx * gridDim.x * blockDim.x;

  decode(mDIdx, dataIdx);

  if(depletionIndicator){

  }
  else{
        // starting the brute force algorithm on q directly
        for ( size_t q = 0; q < k + 1; ++q){
              storageToday = checkStorage(mDIdx);
              tempresult = stateValue( mDIdx[0], 
                                       storageToday, 
                                       0,  q, 
                                       d_randomTable
                                      );
          
              if (tempresult > bestresult){
                bestresult = tempresult;
                //bestq = q;
              }
    
        }
        d_tempTable[dataIdx] = bestresult; // the corresponding q stores in the bestq

  }

}
/* update the value table for one day */
/* only need to hold 2 tables and update each one at a time */
void valueTableUpdateWithPolicy( float** d_valueTables, 
                                 size_t currentTableIdx, 
                                 size_t depletionIndicator,       // either zero or the expected demand for one day
                                 float * d_randomTable,
                                 cudaInfoStruct * cudainfo ){
// each thread will take care of a state at once
  size_t batchAmount = valueTablesLength / cudainfo->numBlocks / cudainfo->numThreadsPerBlock + 1;
  size_t cursor = 0; // holding the index of the next element to be sent to the kernel
  for ( size_t i = 0; i < batchAmount; ++i){
    kernel_valueTableUpdateWithPolicy<<<cudainfo->numBlocks, cudainfo->numThreadsPerBlock>>>(d_randomTable, 
                                                                                              d_valueTables[1 - currentTableIdx], 
                                                                                              d_valueTables[currentTableIdx], 
                                                                                              depletionIndicator,
                                                                                              i);
    cursor += cudainfo->numBlocks * cudainfo->numThreadsPerBlock;
  }

  return;

}

void readfromDevice(float * h_array, float* d_array, size_t length){
        srand(time(0));
        for (size_t i = 0; i < valueTablesLength; ++i){
          h_array[i] = rand() * k * c /RAND_MAX;
        }
}


/* write in the values of the last day in the period */

__global__
void kernel_presetValueTable(float * d_valueTable, size_t table_length){
  extern __shared__ size_t mDIdx[];
  size_t stepSize = gridDim.x * blockDim.x;  // the total number of threads which have been assigned for this task, oneD layout everywhere
  size_t myStartIdx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = myStartIdx; i < table_length; i += stepSize){
    decode(mDIdx, i);
    d_valueTable[i] = checkStorage(mDIdx) * s;
  }

  __syncthreads(); 
}

/* the interface for the main function */
void presetValueTable(float * d_valueTable, size_t  table_length, cudaInfoStruct * cudainfo){
  dim3 gridSize(cudainfo->numBlocks, 1, 1);
  dim3 blockSize(cudainfo->numThreadsPerBlock, 1, 1);
  kernel_presetValueTable<<<gridSize, blockSize, m * sizeof(size_t) >>> ( d_valueTable, table_length);
  return;
}

/* Gather the system information, fol auto fill in the block number and the thread number per block */
void gatherSystemInfo(cudaInfoStruct * cudainfo){

  cudaGetDeviceCount((int*)&(cudainfo->deviceCount));

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  (cudainfo->numThreadsPerBlock) = deviceProp.maxThreadsPerBlock;
  (cudainfo->numBlocks) = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;                                                     //if (T > 1) {cout << " CUDA error at /home/ramp/huangzonghao/ielm2/support.cu:104 code=77(cudaErrorIllegalAddress) exit : fatal error " << endl; exit(-1); };


  return;

}

__global__
void kernel_test(size_t i, size_t * list){
  size_t num = 0;
  if ( threadIdx.x == 1){

      encode(list, &num);
      printf("the result from kernel one is : %d \n", num);
      printf(" now printing the d_list with number m : %d \n",m);
      for (int i = 0; i < m; ++i){
        printf("<%d>", list[i]);
      }
      printf("\n");
  }
}
void test(){

  size_t  ** h_list;
  h_list = (size_t **)malloc(2 * sizeof(size_t *));
  h_list[0] = (size_t * )malloc(20 * sizeof(size_t));
  h_list[1] = (size_t * )malloc(20 * sizeof(size_t));
  // float ** h_list;
  // h_list = (float **)malloc(2 * sizeof(float *));
  // h_list[0] = (float * )malloc(20 * sizeof(float));
  // h_list[1] = (float * )malloc(20 * sizeof(float));

  for (int i = 0; i < m; ++i){
    h_list[0][i] = i * 2 + 1;
  }
  size_t * d_list;
  checkCudaErrors(cudaMalloc(&d_list, m * sizeof(size_t)));

   // passToDevice(h_list[0], d_list, m);
  checkCudaErrors(cudaMemcpy(d_list, h_list[0], m * sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( h_list[1], d_list, m * sizeof(size_t), cudaMemcpyDeviceToHost));

   // readFromDevice(h_list[1], d_list, m);
   cout << endl << "this is the h_list1" << endl;
   for (int i = 0; i < m ; ++i){
     cout << h_list[0][i] << " ";
   }
      cout << endl << "this is the h_list2" << endl;
   for (int i = 0; i < m ; ++i){
     cout << h_list[1][i] << " ";
   }
   cout << endl;

  kernel_test<<<1, 1024>>>(77, (size_t* )d_list);


// test for one d to m d 

  return;
}



