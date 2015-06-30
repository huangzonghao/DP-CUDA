#ifndef _PARA_H
#define _PARA_H
/* the parameters of the program */
/* total number of categories */
static size_t h_m;
static size_t * d_m;

/* maximum number for each category */
static size_t h_k;
static size_t * d_k;

/* number of periods */
static size_t h_T;
static size_t * d_T;

/* storing cost for each item */
static float h_h;
static float * d_h;

/* the price of each item */
static float h_r;
static float * d_r;

/* the ordering cost of each item */
static float h_c;
static float * d_c;

/* the disposal cost of each item */
static float h_theta;
static float * d_theta;

/* the salvage benefit for one item */
static float h_s;
static float * d_s;

/* the discount rate */
static float h_alpha;
static float * d_alpha;

/* maximum storage */
static size_t h_maxhold;
static size_t * d_maxhold;

/* the arrival rate for Poisson distribution */
static float h_lambda;
static float * d_lambda;

static size_t h_min_demand;
static size_t * d_min_demand;

static size_t h_max_demand;
static size_t * d_max_demand;

static float *h_demand_distribution;


#endif
