sed -i "10s/.*/#define ATTENTION_METHOD llm_mha_flatten_attention/" add_dramsyslib_patches/flex_cluster_sdk/test/src/test.c
make iter run