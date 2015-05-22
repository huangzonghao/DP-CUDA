#ifndef _MODEL_H
#define _MODEL_H
void valueTableUpdateWithPolicy( float** d_valueTables, 
                                 size_t currentTableIdx, 
                                 size_t depletionIndicator,       // either zero or the expected demand for one day
                                 float * d_randomTable,
                                 cudaInfoStruct * cudainfo );


void presetValueTable(float * d_valueTable, unsigned long  table_length, cudaInfoStruct * );

#endif
