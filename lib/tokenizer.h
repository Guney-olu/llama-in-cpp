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

#endif
