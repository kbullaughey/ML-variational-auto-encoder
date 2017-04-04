# Variational auto-encoder

Here I reproduce the MNIST variation auto-encoder from:

[Kingma and Welling (2013) Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## Setup

If you need the MNIST data, you can get it already packaged for torch using the script:

    ./get_mnist.sh

I typically use Torch compiled for Lua 5.2 instead of LuaJIT, and prefer to invoke scripts as follows:

    lua -ltorch -lenv <filename.lua>

And include a script `lorch` for this purpose, which allows one to use hash-bang in a portable way:

    #!/usr/bin/env lorch

## Training

At the moment, the script is hard-coded to train a model corresponding to the upper left corner of Figure 2 in the paper (AEVB train).

    ./auto-encoder.lua
