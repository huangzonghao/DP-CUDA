#include "main.h"
#include <cmath>
using namespace std;

void printResult(){
    
}

int main(){
	/* declare variables */
    /* system features */
    cudaInfoStruct cudainfo;
    gatherSystemInfo(&cudainfo);,

	// host tables
	float * h_valueTable;

	// device tables
	// two tables in total
	float ** d_valueTables[2];
    unsigned long  valueTablesLength;
    valueTablesLength = pow(k, m);

	/* memory allocation */
	h_valueTable = malloc( valueTablesLength * sizeof(float) );
    deviceTableInit(2, 					// total number of tables
    				d_valueTables, 		
    				valueTablesLength,
                    &cudainfo
    				);


    /* evalueate the value of the table with the given parameters and the given policy f */
    evalWithPolicy( h_valueTable, d_valueTables, valueTablesLength, &cudainfo);

    /* output the file */
    printResult();


    return 0;
}

