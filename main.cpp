#include "lib/build_transformer.h"


int main() {
    const char* checkpoint_path = "./llama-in-cpp/models/stories15M.bin";

    Transformer transformer;

    build_transformer(transformer, checkpoint_path);
}