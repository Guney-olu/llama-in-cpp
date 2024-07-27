# Loading llama weights using Pure C++

```bash
g++ -o transformer llama-run.cpp -std=c++11
./transformer
```
## unsorted todos

- Add tokenizer code to generate something
- Add support for safetensors format file[*]
- Add more tests and model support
- Bechmarking fo toks and comparision

**Inspired by do check this out -> https://github.com/karpathy/llama2.c**