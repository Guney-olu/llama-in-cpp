/* 
TODO
Fixing the main issues and making it more readable
*/

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <cmath>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

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

#include "/usr/local/Cellar/libomp/18.1.8/include/omp.h"

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


struct TokenIndex {
    char *str;
    int id;
};

struct Tokenizer {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
};

int compare_tokens(const void* a, const void* b) {
    return std::strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;

    t->vocab = (char**)std::malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)std::malloc(vocab_size * sizeof(float));
    t->sorted_vocab = nullptr; 

    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = static_cast<unsigned char>(i);
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    std::ifstream file(tokenizer_path, std::ios::binary);
    if (!file) {
        std::cerr << "couldn't load " << tokenizer_path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (!file.read(reinterpret_cast<char*>(&t->max_token_length), sizeof(int))) {
        std::cerr << "failed read" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (!file.read(reinterpret_cast<char*>(&t->vocab_scores[i]), sizeof(float))) {
            std::cerr << "failed read" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        if (!file.read(reinterpret_cast<char*>(&len), sizeof(int))) {
            std::cerr << "failed read" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        t->vocab[i] = (char*)std::malloc(len + 1);
        if (!file.read(t->vocab[i], len)) {
            std::cerr << "failed read" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0'; 
    }
    file.close();
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char* piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (std::sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = reinterpret_cast<char*>(t->byte_pieces) + byte_val * 2;
    }
    return piece;
}

void safe_printf(const char* piece) {
    if (piece == nullptr || piece[0] == '\0') { 
        return; 
    }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(std::isprint(byte_val) || std::isspace(byte_val))) {
            return; 
        }
    }
    std::printf("%s", piece);
}

int str_lookup(const char* str, TokenIndex* sorted_vocab, int vocab_size) {
    TokenIndex tok = { const_cast<char*>(str), 0 };
    TokenIndex* res = reinterpret_cast<TokenIndex*>(std::bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens));
    return res != nullptr ? res->id : -1;
}


void encode(Tokenizer* t, const char* text, int8_t bos, int8_t eos, int* tokens, int* n_tokens) {

    if (text == nullptr) { 
        std::fprintf(stderr, "cannot encode NULL text\n"); 
        std::exit(EXIT_FAILURE); 
    }

    if (t->sorted_vocab == nullptr) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = reinterpret_cast<TokenIndex*>(std::malloc(t->vocab_size * sizeof(TokenIndex)));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        std::qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    std::vector<char> str_buffer(t->max_token_length * 2 + 1 + 2);
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }


    for (const char* c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }

        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // Ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer.data(), t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // We found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
 
            for (size_t i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = static_cast<unsigned char>(str_buffer[i]) + 3;
            }
        }
        str_len = 0; 
    }

    while (true) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            std::sprintf(str_buffer.data(), "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer.data(), t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; 
        }

        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--; 
    }

    if (eos) tokens[(*n_tokens)++] = 2;
}

struct ProbIndex {
    float prob;
    int index;
}; 

struct Sampler {
    int vocab_size;
    ProbIndex* probindex; 
    float temperature;
    float topp;
    unsigned long long rng_state;
};

int sample_argmax(const float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(const float* probabilities, int n, float coin) {

    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; 
}

bool compare(const ProbIndex& a, const ProbIndex& b) {
    return a.prob > b.prob;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;

    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }

    std::sort(probindex, probindex + n0, compare);


    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; 
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = new ProbIndex[sampler->vocab_size];
}

void free_sampler(Sampler* sampler) {
    delete[] sampler->probindex;
}

unsigned int random_u32(unsigned long long* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return static_cast<unsigned int>((*state * 0x2545F4914F6CDD1Dull) >> 32);
}

float random_f32(unsigned long long* state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q = 0; q < sampler->vocab_size; q++) {
            logits[q] /= sampler->temperature;
        }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

#include <chrono>
long time_in_ms() {
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
    return duration.count();
}


void generate(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, char* prompt, int steps) {
    const char* empty_prompt = "";
    if (prompt == nullptr) {
        prompt = const_cast<char*>(empty_prompt);
    }

    int num_prompt_tokens = 0;
    int* prompt_tokens = new int[strlen(prompt) + 3]; // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        std::cerr << "something is wrong, expected at least 1 prompt token" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    long start = 0; // used to time our code, only initialized after first iteration
    int next;       // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;    // position in the sequence
    while (pos < steps) {

        float* logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        if (next == 1) {
            break;
        }

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); 
        std::cout.flush();
        token = next;

        if (start == 0) {
            start = time_in_ms();
        }
    }
    std::cout << std::endl;

    if (pos > 1) {
        long end = time_in_ms();
        std::cerr << "achieved tok/s: " << (pos - 1) / (double)(end - start) * 1000 << std::endl;
    }

    delete[] prompt_tokens;
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    std::cout << guide;
    if (fgets(buffer, bufsize, stdin) != nullptr) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}



int main() {
    const char* checkpoint_path = "./llama-in-cpp/models/stories15M.bin";
    const char* tokenizer_path = "./llama-in-cpp/models/tok512.bin";
    float temperature = 1.0f;  
    float topp = 0.9f;  
    int steps = 256; 
    char *prompt = NULL;
    unsigned long long rng_seed = 0;


    Transformer transformer;

    build_transformer(transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length


    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    generate(&transformer, &tokenizer, &sampler, prompt, steps);
    
    
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(transformer);
    return 0;
}
