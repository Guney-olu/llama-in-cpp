#include "lib/build_transformer.h"
#include "lib/tokenizer.h"

int main() {
    const char* checkpoint_path = "./llama-in-cpp/models/stories15M.bin";
    const char* tokenizer_path = "./llama-in-cpp/models/tokenizer.bin";

    Transformer transformer;

    build_transformer(transformer, checkpoint_path);


    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    
    free_tokenizer(&tokenizer);
    
    free_transformer(transformer);
}