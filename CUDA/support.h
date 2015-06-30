#ifndef _SUPPORT_H
#define _SUPPORT_H

struct cudaInfoStruct{
  size_t deviceCount;
  size_t numBlocks;
  size_t numThreadsPerBlock;
};


void readFromDevice(float * h_array, float* d_array, size_t length);
void passToDevice(float* h_array, float* d_array, size_t length);
void passToDevice(const float* h_array, float* d_array, size_t length);
void readfromDevice(float * h_array, float* d_array, size_t length);

void gatherSystemInfo(cudaInfoStruct *);



void deviceTableInit(size_t numTables, float ** tables, unsigned long tableLengths, cudaInfoStruct * cudainfo);
void test();


/* the device functions */
// __device__ void decode (size_t* mDIdx, size_t oneDIdx);
// __device__ size_t checkStorage(size_t* mDarray);


#endif
