#ifndef HELPER_H
#define HELPER_H

#include "/usr/local/Cellar/libomp/18.1.8/include/omp.h"
#include <cmath>

void rmsnorm(float* o, float* x, const float* weight, int size){
    float ss =0.0f;
    for (int j=0; j<size; j++){
        ss += x[j] * x[j];
    }
    ss = ss/size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    } 
}

void softmax(float* x, int size) {
    // Finding the max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}


void matmul(float* xout, const float* x, const float* w, int n, int d) {
    // W (d, n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    #pragma omp parallel for
    for (int i = 0; i < d; ++i) {
        float val = 0.0f;
        for (int j = 0; j < n; ++j) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}



#endif
