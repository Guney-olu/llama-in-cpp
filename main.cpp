#include "lib/build_transformer.h"
#include "lib/tokenizer.h"

int main() {
    const char* checkpoint_path = "./llama-in-cpp/models/pytorch_model.bin";
    const char* tokenizer_path = "./llama-in-cpp/models/tokenizer.bin";


    float temperature = 1.0f;  
    float topp = 0.9f;  
    int steps = 256; 
    char *prompt = "tell the story";
    unsigned long long rng_seed = 0;


    Transformer transformer;

    build_transformer(transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length


    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    generate(&transformer, &tokenizer, &sampler, prompt, steps);
    
    free_tokenizer(&tokenizer);
    
    free_transformer(transformer);
}