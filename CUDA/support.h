#ifndef _SUPPORT_H
#define _SUPPORT_H
#include "timer.h"
#include "parameters.h"
#include "utils.h"

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif



#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

struct cudaInfoStruct{
  size_t deviceCount = 0;
  size_t numBlocks = 0;
  size_t numThreadsPerBlock = 0;
};

void valueTableUpdate( float* d_randomTable,
                       float d_valueTable, 
                       const size_t m, 
                       const size_t k,
                       const size_t numSamples,
                       const float holdingCost,
                       const float sellingPrice,
                       const float orderingCost,
                       const float disposalCost,
                       const float salvageValue,
                       const float discountRate,
                       const size_t maxHold
                       const dim3 kernelSize,       // number of blocks launched 
                       const dim3 blockSize );     // number of threads per block launched


#endif
