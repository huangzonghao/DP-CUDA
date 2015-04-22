#include "main.h"
#include <cmath>
using namespace std;

unsigned long  valueTablesLength;

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
	// two tables in total for cross updating and another table to store the random distribution
	float ** d_valueTables[2];
    float * d_randomTable;
    
    valueTablesLength = pow(k, m);

	/* memory allocation */
	h_valueTable = malloc( valueTablesLength * sizeof(float) );
    deviceTableInit(2, 					// total number of tables
    				d_valueTables, 		
    				valueTablesLength,
                    &cudainfo
    				);
    deviceTableInit(1,
                    d_randomTable,
                    (max_demand - min_demand) * sizeof(float),
                    &cudainfo);
    passToDevice(demand_distribution, d_randomTable, (max_demand - min_demand) * sizeof(float));


    /* evalueate the value of the table with the given parameters and the given policy f */
    // the policy is : the depletion amount is always zero except the first day
    size_t currentTableIdx = 0;      // the index of table to be updated in next action

    /*first init make one of the d_valueTables to the edge values*/
    presetValueTable(d_valueTables[currentTableIdx], valueTablesLength, cudainfo);
    currentTableIdx = 1 - currentTableIdx;

    /* T periods in total */
    for ( size_t iPeriod = T; iPeriod > 0; --iPeriod){
      if(iPeriod != 1){
        valueTableUpdateWithPolicy( d_valueTables, currentTableIdx, 0, d_randomTable, cudainfo);
        currentTableIdx = 1 - currentTableIdx;
      }
      else{
        // first calculate the expect demand for each day
        float expectDemand = 0;
        for (int i = min_demand; i < max_demand - min_demand + 1; ++i){
            expectDemand += i * demand_distribution[i];
        }
        valueTableUpdateWithPolicy( d_valueTables, currentTableIdx, (size_t)ceil(expectDemand), d_randomTable, cudainfo);
      }
    }
    
    // the final result stores in the (1 - currentTableIdx)
    readFromDevice(h_valueTable, d_valueTables[1 - currentTableIdx], tableLength);

    /* output the file */
    printResult();


    return 0;
}

