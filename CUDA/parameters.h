#ifndef _PARA_H
#define _PARA_H
/* the parameters of the program */
/* total number of categories */
size_t m;

/* maximum number for each category */
size_t k;

/* number of periods */
size_t T;

/* storing cost for each item */
float h;

/* the price of each item */
float r;

/* the ordering cost of each item */
float c;

/* the disposal cost of each item */
float theta;

/* the salvage benefit for one item */
float s;

/* the discount rate */
float alpha;

/* maximum storage */
size_t maxhold;

/* the arrival rate for poisson distribution */
float lambda;

size_t min_demand;
size_t max_demand;

float *demand_distribution;

#endif
