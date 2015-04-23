#ifndef _SUPPORT_H
#define _SUPPORT_H
#include "timer.h"
#include "parameters.h"
#include <cstdlib>
#include <ctime>
//#include "utils.h"


struct cudaInfoStruct{
  size_t deviceCount;
  size_t numBlocks;
  size_t numThreadsPerBlock;
};


void readFromDevice(float * h_array, float* d_array, size_t length);
void passToDevice(float* h_array, float* d_array, size_t length);
void passToDevice(const float* h_array, float* d_array, size_t length);
void readfromDevice(float * h_array, float* d_array, size_t length);

void presetValueTable(float * d_valueTable, unsigned long  table_length, cudaInfoStruct * );
void gatherSystemInfo(cudaInfoStruct *);

void valueTableUpdateWithPolicy( float** d_valueTables, 
                                 size_t currentTableIdx, 
                                 size_t depletionIndicator,       // either zero or the expected demand for one day
                                 float * d_randomTable,
                                 cudaInfoStruct * cudainfo );

void deviceTableInit(size_t numTables, float ** tables, unsigned long tableLengths, cudaInfoStruct * cudainfo);
void test();


#endif
