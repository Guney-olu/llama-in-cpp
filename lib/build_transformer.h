#ifndef CHECKPOINT_READER_H
#define CHECKPOINT_READER_H

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <cmath>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include "helpers.h"
#include "/usr/local/Cellar/libomp/18.1.8/include/omp.h"


using namespace std;

struct Config {
    int dim;
    int hidden_dim; 
    int n_layers; 
    int n_heads;
    int n_kv_heads; 
    int vocab_size;
    int seq_len; 
};

struct TransformerWeights {
    float* token_embedding_table;
    float* rms_att_weight;
    float* rms_ffn_weight;

    float* wq;
    float* wk;
    float* wv;
    float* wo;

    float* w1;
    float* w2;
    float* w3;

    float* rms_final_weight;
    float* wcls;
};

struct State {
    float* x;
    float* xb;
    float* xb2;
    float* hb;
    float* hb2;
    float* q;
    float* k;
    float* v;
    float* att;
    float* logits;
    // kv cache - -
    float* key_cache;
    float* value_cache;
};

struct Transformer {
    Config config;
    TransformerWeights weights;
    State state;
    int fd;
    float* data;
    ssize_t file_size;
};

void malloc_run_state(State& s, const Config& p) {
    int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    s.x = static_cast<float*>(calloc(p.dim, sizeof(float)));
    s.xb = static_cast<float*>(calloc(p.dim, sizeof(float)));
    s.xb2 = static_cast<float*>(calloc(p.dim, sizeof(float)));
    s.hb = static_cast<float*>(calloc(p.hidden_dim, sizeof(float)));
    s.hb2 = static_cast<float*>(calloc(p.hidden_dim, sizeof(float)));
    s.q = static_cast<float*>(calloc(p.dim, sizeof(float)));
    s.key_cache = static_cast<float*>(calloc(p.n_layers * p.seq_len * kv_dim, sizeof(float)));
    s.value_cache = static_cast<float*>(calloc(p.n_layers * p.seq_len * kv_dim, sizeof(float)));
    s.att = static_cast<float*>(calloc(p.n_heads * p.seq_len, sizeof(float)));
    s.logits = static_cast<float*>(calloc(p.vocab_size, sizeof(float)));

    // Checking all allocations
    if (!s.x || !s.xb || !s.xb2 || !s.hb || !s.hb2 || !s.q || !s.key_cache || !s.value_cache || !s.att || !s.logits) {
        cerr << "malloc failed!" << endl;
        exit(EXIT_FAILURE);
    }
}

void free_run_state(State& s) {
    free(s.x);
    free(s.xb);
    free(s.xb2);
    free(s.hb);
    free(s.hb2);
    free(s.q);
    free(s.att);
    free(s.logits);
    free(s.key_cache);
    free(s.value_cache);
}

void memory_map_weights(TransformerWeights& w, const Config& p, float* ptr, int shared_weights) {
    int head_size = p.dim / p.n_heads;
    unsigned long long n_layers = p.n_layers;
    w.token_embedding_table = ptr;
    ptr += p.vocab_size * p.dim;
    w.rms_att_weight = ptr;
    ptr += n_layers * p.dim;
    w.wq = ptr;
    ptr += n_layers * p.dim * (p.n_heads * head_size);
    w.wk = ptr;
    ptr += n_layers * p.dim * (p.n_kv_heads * head_size);
    w.wv = ptr;
    ptr += n_layers * p.dim * (p.n_kv_heads * head_size);
    w.wo = ptr;
    ptr += n_layers * (p.n_heads * head_size) * p.dim;
    w.rms_ffn_weight = ptr;
    ptr += n_layers * p.dim;
    w.w1 = ptr;
    ptr += n_layers * p.dim * p.hidden_dim;
    w.w2 = ptr;
    ptr += n_layers * p.hidden_dim * p.dim;
    w.w3 = ptr;
    ptr += n_layers * p.dim * p.hidden_dim;
    w.rms_final_weight = ptr;
    ptr += p.dim;
    ptr += p.seq_len * head_size / 2;
    ptr += p.seq_len * head_size / 2; 
    w.wcls = shared_weights ? w.token_embedding_table : ptr;
}

void read_checkpoint(const char* checkpoint, Config& config, TransformerWeights& weights, int& fd, float*& data, ssize_t& file_size) {
    ifstream file(checkpoint, ios::binary);
    if (!file.is_open()) {
        cerr << "Couldn't open file which contain weights" << checkpoint << endl;
        exit(EXIT_FAILURE);
    }

    if (!file.read(reinterpret_cast<char*>(&config), sizeof(Config))) {
        cerr << "Failed to read config" << endl;
        exit(EXIT_FAILURE);
    }

    int shared_weights = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size = abs(config.vocab_size);

    file.seekg(0, ios::end);
    file_size = file.tellg();
    file.close();

    fd = open(checkpoint, O_RDONLY);
    if (fd == -1) {
        cerr << "open failed!!" << endl;
        exit(EXIT_FAILURE);
    }

    data = static_cast<float*>(mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (data == MAP_FAILED) {
        cerr << "mmap failed!" << endl;
        exit(EXIT_FAILURE);
    }

    float* weights_ptr = data + sizeof(Config) / sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
    for (int i = 0; i < 10; ++i) {
        std::cout << "Weight " << i << ": " << weights_ptr[i] << std::endl;
    }
}

void build_transformer(Transformer& t, const char* checkpoint_path) {
    
    read_checkpoint(checkpoint_path, t.config, t.weights, t.fd, t.data, t.file_size);
    malloc_run_state(t.state, t.config);
}

void free_transformer(Transformer& t) {
    if (t.data != MAP_FAILED) {
        munmap(t.data, t.file_size);
    }
    if (t.fd != -1) {
        close(t.fd);
    }
    free_run_state(t.state);
}


//Doing the forward pass
float* forward(Transformer* transformer, int token, int pos) {

    // Convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    State* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(float));

    // Forwarding  all the layers
    for (unsigned long long l = 0; l < p->n_layers; l++) {

        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        int loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / std::pow(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = std::cos(val);
            float fci = std::sin(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
        #pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= std::sqrt(head_size);
                att[t] = score;
            }
            softmax(att, pos + 1);
            float* xb = s->xb + h * head_size;
            std::memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        // Applying SwiGLU 
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= 1.0f / (1.0f + std::exp(-val));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }
    rmsnorm(x, x, w->rms_final_weight, dim);
    // Classification into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}





#endif
