# Loading llama weights using Pure C++

**Setup the Path for yout model.bin and tokeiner in the llama-run.cpp**

```bash
g++ -o transformer deprecated/llama-run.cpp -std=c++11 
./transformer
```

```cpp

#include "lib/build_transformer.h"
#include "lib/tokenizer.h"

const char* checkpoint_path = "./llama-in-cpp/models/pytorch_model.bin";
Transformer transformer;
//Loading weights
build_transformer(transformer, checkpoint_path);

```
**Check main.cpp to to run the tinystory example**

## unsorted todos

- Add tokenizer code to generate something
- Add support for safetensors format file[*]
- Add more tests and model support
- Bechmarking fo toks and comparision

**Inspired by do check this out -> https://github.com/karpathy/llama2.c**