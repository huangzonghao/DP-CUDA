#include "main.h"
#include <cmath>
using namespace std;

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
    size_t * valueTablesLength[2];
    valueTablesLength[0] = pow(k, m);
    valueTablesLength[1] = valueTablesLength[0];

	/* memory allocation */
	h_valueTable = malloc( pow(k, m) * sizeof(float) );
    deviceTableInit(2, 					// total number of tables
    				d_valueTables, 		
    				valueTablesLength,
                    &cudainfo
    				);


    /* evalueate the value of the table with the given parameters and the given policy f */
    evalWithPolicy( h_valueTable, d_valueTables, &cudainfo);


        return 0;
}

