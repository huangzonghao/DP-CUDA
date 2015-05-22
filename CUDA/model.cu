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

/* evaluate the state value given z and q */
/* return the expected value over the demands */
// i don't have to get all the storage information to get the state value of today
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
                                         size_t* d_mdidx,
                                         size_t depletionIndicator,
                                         size_t valueTablesLength,
                                         size_t batchIdx
                                         ){
  float bestresult = 0;
  //float bestq = 0;
  float tempresult = 0;
  size_t storageToday = 0;
  // this is both the thread index and the data index in this batch
  size_t myIdx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t dataIdx = myIdx + batchIdx * gridDim.x * blockDim.x;

  // size_t testnum = 1601;
  // if(dataIdx == testnum){
  //   printf(" \n Now printing the calculation of the entry \n %d \n", dataIdx);
  // }

  if(dataIdx < valueTablesLength){

          decode(&d_mdidx[myIdx * m], dataIdx);

          if(depletionIndicator){
                  storageToday = checkStorage(&d_mdidx[myIdx * m]);

                for ( size_t q = 0; q < k; ++q){
                      tempresult = stateValue( d_mdidx[myIdx * m], 
                                               storageToday, 
                                               depletionIndicator * T,  q, 
                                               d_randomTable
                                              );

                      if (tempresult > bestresult){
                        bestresult = tempresult;
                        //bestq = q;
                      }
            
                }

                d_tempTable[dataIdx] = bestresult;

          }
          else{
                // starting the brute force algorithm on q directly
                  storageToday = checkStorage(&d_mdidx[myIdx * m]);

                  // if( dataIdx == testnum){
                  //   printf("\n storage today : %d ", storageToday);
                  //   printf("\n expiring today : %d ", d_mdidx[myIdx * m]);
                  // }

                for ( size_t q = 0; q < k; ++q){
                      tempresult = stateValue( d_mdidx[myIdx * m], 
                                               storageToday, 
                                               0,  q, 
                                               d_randomTable
                                              );
                  // if( dataIdx == testnum){
                  //   printf("\n tempresult  <%d> : %f",q, tempresult);
                  // }
                  
                      if (tempresult > bestresult){
                        bestresult = tempresult;
                        //bestq = q;
                      }
            
                }
                  // if( dataIdx == testnum){
                  //   printf("\n");
                  // }
                d_tempTable[dataIdx] = bestresult; // the corresponding q stores in the bestq
          }
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
  
  size_t * d_mdidx; 
  // assign to each thread some global memory to store the m D information
  checkCudaErrors(cudaMalloc(&d_mdidx, cudainfo->numBlocks * cudainfo->numThreadsPerBlock * m * sizeof(size_t)));


  size_t batchAmount = valueTablesLength / cudainfo->numBlocks / cudainfo->numThreadsPerBlock + 1;

  for ( size_t i = 0; i < batchAmount; ++i){

    kernel_valueTableUpdateWithPolicy<<<cudainfo->numBlocks, cudainfo->numThreadsPerBlock>>>(d_randomTable, d_valueTables[1 - currentTableIdx], d_valueTables[currentTableIdx], d_mdidx,depletionIndicator, valueTablesLength, i);
  }


  checkCudaErrors(cudaFree(d_mdidx));
  return;
}


/* write in the values of the last day in the period */

// note the shared memory is shared among all the threads and has a limit per block
__global__
void kernel_presetValueTable(float * d_valueTable, size_t * d_mdidx, size_t table_length){

  size_t stepSize = gridDim.x * blockDim.x;  // the total number of threads which have been assigned for this task, oneD layout everywhere
  size_t myStartIdx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = myStartIdx; i < table_length; i += stepSize){
    decode(&d_mdidx[myStartIdx * m], i);
    d_valueTable[i] = checkStorage(&d_mdidx[myStartIdx * m]) * s;
  }
  __syncthreads(); 
}

/* the interface for the main function */
void presetValueTable(float * d_valueTable, size_t  table_length, cudaInfoStruct * cudainfo){
  dim3 gridSize(cudainfo->numBlocks, 1, 1);
  dim3 blockSize(cudainfo->numThreadsPerBlock, 1, 1);
  size_t * d_mdidx; 
  // assign to each thread some global memory to store the m D information
  checkCudaErrors(cudaMalloc(&d_mdidx, cudainfo->numBlocks * cudainfo->numThreadsPerBlock * m * sizeof(size_t)));
  kernel_presetValueTable<<<gridSize, blockSize>>> ( d_valueTable, d_mdidx, table_length);
  checkCudaErrors(cudaFree(d_mdidx));
  return;
}