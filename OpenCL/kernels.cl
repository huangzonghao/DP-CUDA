__kernel void keDeviceTableInit(float* d_valueTable, size_t arrayLength  ){
  size_t stepSize = gridDim.x * blockDim.x;  // the total number of threads which have been assigned for this task
  size_t myStartIdx = blockDim.x * blockIdx.x + threadIdx.x;
  for (size_t i = myStartIdx; i < arrayLength; i += stepSize)
    d_valueTable[i] = 0;

}

// __kernel void decode (size_t* mDIdx, size_t oneDIdx, size_t * d_m, size_t * d_k){

//       for( size_t i = 0; i < *d_m ; ++i){
//             mDIdx[*d_m - 1 - i] = oneDIdx % *d_k;
//             oneDIdx /= *d_k;
//       }

// }

// __kernel size_t checkStorage(size_t* mDarray, size_t * d_m){
//     size_t result = 0;
//     for (size_t i = 0; i < *d_m ; ++i ){
//             result += mDarray[i];
//     }
//     return result;
// }

// __kernel float stateValue( size_t expiringToday,
//                   int storageToday,
//                   int z, int q,
//                   float * d_randomTable,
//                   // the parameters
//                   size_t * d_min_demand,
//                   size_t * d_max_demand,
//                   float * d_s,
//                   float * d_h,
//                   float * d_alpha,
//                   float * d_c,
//                   float * d_r,
//                   float * d_theta,
//                   size_t * d_m,
//                   size_t * d_k
//                   ){

//         float profit = 0;
//         float sum = 0;
//         for ( size_t i = *d_min_demand; i < *d_max_demand; ++i){
//                 profit = *d_s * z     \                                                        // the money collected depletion
//                        - *d_h * max(int(int(storageToday) - z) , 0)   \                             // the cost for holding all the items
//                        - *d_alpha * *d_c * q   \                                                    // the money spent on ordering new items
//                        + *d_alpha * *d_r * min(int(i), int(storageToday - z + q))    \           // the total income from selling the products to the customers
//                        - *d_alpha * *d_theta * max(int(expiringToday - z - i), 0);// the money spent on the expired items

//                 sum += profit * d_randomTable[i];
//         }

//         return sum;

// }


__kernel void keValueTableUpdateWithPolicy( float* d_randomTable,
                                        float* d_valueTable,     // note both the value table and the temp table here hold the exact starting index for this kernel launch
                                        float* d_tempTable,
                                        size_t* d_mdidx,
                                        size_t depletionIndicator,
                                        size_t valueTablesLength,
                                        size_t batchIdx,

                                        // parameters
                                        size_t * d_min_demand,
                                        size_t * d_max_demand,
                                        float * d_s,
                                        float * d_h,
                                        float * d_alpha,
                                        float * d_c,
                                        float * d_r,
                                        float * d_theta,
                                        size_t * d_m,
                                        size_t * d_k,
                                        size_t * d_T
                                         ){

  for ( int iT = d_T; iT > 0; iT--){

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

          decode(&d_mdidx[myIdx * *d_m], dataIdx, d_m, d_k);

          if(depletionIndicator){
                  storageToday = checkStorage(&d_mdidx[myIdx * *d_m], d_m);

                for ( size_t q = 0; q < *d_k; ++q){
                      tempresult = stateValue( d_mdidx[myIdx * *d_m],
                                               storageToday,
                                               depletionIndicator * *d_T,  q,
                                               d_randomTable,

                                               d_min_demand,
                                               d_max_demand,
                                               d_s,
                                               d_h,
                                               d_alpha,
                                               d_c,
                                               d_r,
                                               d_theta,
                                               d_m,
                                               d_k
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
                  storageToday = checkStorage(&d_mdidx[myIdx * *d_m], d_m);

                  // if( dataIdx == testnum){
                  //   printf("\n storage today : %d ", storageToday);
                  //   printf("\n expiring today : %d ", d_mdidx[myIdx * m]);
                  // }

                for ( size_t q = 0; q < *d_k; ++q){
                      tempresult = stateValue( d_mdidx[myIdx * *d_m],
                                               storageToday,
                                               0,  q,
                                               d_randomTable,

                                               d_min_demand,
                                               d_max_demand,
                                               d_s,
                                               d_h,
                                               d_alpha,
                                               d_c,
                                               d_r,
                                               d_theta,
                                               d_m,
                                               d_k
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

}



/* write in the values of the last day in the period */

// note the shared memory is shared among all the threads and has a limit per block
__kernel void kePresetValueTable( float * d_valueTable,
                              size_t * d_mdidx,
                              size_t table_length,
                              // parameters
                              size_t * d_m,
                              size_t * d_k,
                              float * d_s
                            ){

  size_t stepSize = gridDim.x * blockDim.x;  // the total number of threads which have been assigned for this task, oneD layout everywhere
  size_t myStartIdx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = myStartIdx; i < table_length; i += stepSize){
    decode(&d_mdidx[myStartIdx * *d_m], i, d_m, d_k);
    d_valueTable[i] = checkStorage(&d_mdidx[myStartIdx * *d_m], d_m) * *d_s;
  }
  __syncthreads();
}


