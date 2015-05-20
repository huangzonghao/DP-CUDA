#include "include/main.h"
#include <cmath>
using namespace std;

size_t  valueTablesLength;
int loadParams(string);

void printResult(float* valueTable, size_t length, string title = "value table"){
    cout << endl << "Now starting to print out the " << title << ": "  << endl;
    // for (size_t i = 0; i < length - 1; ++i){
    //         cout << valueTable[i] << ", ";
    // }
    // cout << valueTable[length - 1] << endl;
    for (size_t i = 0; i < length; ++i){
      cout << " the node : " << i << " has value : " << valueTable[i] << endl;

    }

    return;

}

int main(int argc, char ** argv){
    /* load the global variables */
    if ( argc < 2){
        cerr << " Please input the config file name." << endl;
        return -1;
    }

    loadParams(string(argv[1]));

	/* declare variables */
    /* system features */

    cudaInfoStruct cudainfo;
    cudainfo.deviceCount = 0;
    cudainfo.numBlocks = 0;
    cudainfo.numThreadsPerBlock = 0;
    gatherSystemInfo(&cudainfo);

    cout << "System Configuration : " << endl 
         << "   Number of CUDA Devices : " << cudainfo.deviceCount << endl 
         << "   Number of cores : " << cudainfo.numBlocks << endl 
         << "   Number of threads per core : " << cudainfo.numThreadsPerBlock << endl;

	// host tables
	float * h_valueTable;

	// device tables
	// two tables in total for cross updating and another table to store the random distribution
	float ** d_valueTables;
    d_valueTables = (float **)malloc(2 * sizeof(float *));
    float * d_randomTable;

    
    valueTablesLength = pow(k, m);
    cout <<"valueTablesLength: " <<  valueTablesLength << endl;



	/* memory allocation */
	h_valueTable = (float*)malloc( valueTablesLength * sizeof(float) );

    deviceTableInit(2, 					// total number of tables
    				d_valueTables, 		
    				valueTablesLength,
                    &cudainfo);


    deviceTableInit(1,
                    &d_randomTable,
                    (max_demand - min_demand),
                    &cudainfo);

    passToDevice(demand_distribution, d_randomTable, (max_demand - min_demand));


    /* evalueate the value of the table with the given parameters and the given policy f */
    // the policy is : the depletion amount is always zero except the first day
    size_t currentTableIdx = 0;      // the index of table to be updated in next action

    // /*first init make one of the d_valueTables to the edge values*/
    presetValueTable(d_valueTables[currentTableIdx], valueTablesLength, &cudainfo);
    currentTableIdx = 1 - currentTableIdx;

    readFromDevice(h_valueTable, d_valueTables[1 - currentTableIdx], valueTablesLength);



    // /* T periods in total */
    for ( size_t iPeriod = T; iPeriod > 0; --iPeriod){
      if(iPeriod != 1){
        valueTableUpdateWithPolicy( d_valueTables, currentTableIdx, 0, d_randomTable, &cudainfo);
        currentTableIdx = 1 - currentTableIdx;
      }
      else{
        // first calculate the expect demand for each day
        float expectDemand = 0;
        for (int i = min_demand; i < max_demand - min_demand + 1; ++i){
            expectDemand += i * demand_distribution[i];
        }
                // cerr << " the expected demand : " << endl << (size_t)ceil(expectDemand) << endl;
        valueTableUpdateWithPolicy( d_valueTables, currentTableIdx, (size_t)ceil(expectDemand), d_randomTable, &cudainfo);
      }
    }
    readFromDevice(h_valueTable, d_valueTables[1 - currentTableIdx], valueTablesLength);
    printResult(h_valueTable, valueTablesLength, "Value table");

    return 0;
}

