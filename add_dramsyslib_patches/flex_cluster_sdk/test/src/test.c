#include "flex_runtime.h"
#include "kernels/gemm/gemm_systolic_wise.h"
#include <math.h>
#include "flex_vecteng.h"
#include "flex_mtxtran.h"
#include "flex_redmule.h"
#include "kernels/llm_mha/llm_mha_flash_attention.h"
#include "kernels/llm_mha/llm_mha_flatten_attention.h"

#define ATTENTION_METHOD llm_mha_flatten_attention

int main()
{
    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    flex_timer_start();
    /**************************************/
    /*  Program Execution Region -- Start */
    /**************************************/

    ATTENTION_METHOD(4096, 4096, 128, 32, 8, 2);

    /**************************************/
    /*  Program Execution Region -- Stop  */
    /**************************************/
    flex_global_barrier_xy();
    flex_timer_end();
    flex_eoc(eoc_val);
    return 0;
}