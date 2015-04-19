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

void evalWithPolicy(float* h_valueTable, float * d_valueTables, cudaInfoStruct * cudainfo);
void presetValueTable(float * d_valueTable, unsigned long  table_length, cudaInfoStruct * cudainfo);
void gatherSystemInfo(size_t * deviceCount, size_t * numBlocks, size_t * numThreadsPerBlock);

#endif
