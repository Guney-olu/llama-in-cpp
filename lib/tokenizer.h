#ifndef TOKENZIER
#define TOKENZIER


#include <cstring>
#include <cstdlib>
#include <iostream> 
#include <fstream>


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
        std::cerr << "Couldn't load " << tokenizer_path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (!file.read(reinterpret_cast<char*>(&t->max_token_length), sizeof(int))) {
        std::cerr << "Failed to read max_token_length" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (!file.read(reinterpret_cast<char*>(&t->vocab_scores[i]), sizeof(float))) {
            std::cerr << "Failed to read vocab_scores at index " << i << std::endl;
            std::exit(EXIT_FAILURE);
        }

        if (!file.read(reinterpret_cast<char*>(&len), sizeof(int))) {
            std::cerr << "Failed to read vocab length at index " << i << std::endl;
            std::exit(EXIT_FAILURE);
        }

        t->vocab[i] = (char*)std::malloc(len + 1);
        if (!file.read(t->vocab[i], len)) {
            std::cerr << "Failed to read vocab at index " << i << std::endl;
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


#endif
