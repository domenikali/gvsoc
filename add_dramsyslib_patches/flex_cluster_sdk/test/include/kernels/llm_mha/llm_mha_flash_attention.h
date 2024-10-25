#ifndef _LLM_MHA_FLASH_ATTENTION_H_
#define _LLM_MHA_FLASH_ATTENTION_H_

#include <math.h>
#include "snrt.h"
#include "flex_runtime.h"
#include "flex_vecteng.h"
#include "flex_mtxtran.h"
#include "flex_redmule.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"

void llm_mha_flash_attention_each_head_orignal_in_paper(uint32_t BR, uint32_t BC, uint32_t TR, uint32_t TC, uint32_t D, uint32_t elem_size){
    FlexPosition pos = get_pos(flex_get_cluster_id());
    uint32_t Q_tilesize  = BR*D*elem_size;
    uint32_t K_tilesize  = BC*D*elem_size;
    uint32_t V_tilesize  = BC*D*elem_size;
    uint32_t O_tilesize  = BR*D*elem_size;
    uint32_t m_tilesize  = BR*elem_size;
    uint32_t l_tilesize  = BR*elem_size;

    for (int j = 0; j < TC; ++j)
    {
        //0. Load K and V
        if (flex_is_dm_core())
        {
            flex_dma_1d(local(0),hbm_west(pos.y,0), V_tilesize);
            flex_dma_1d(local(0),hbm_south(pos.x,0), K_tilesize);
        }
        flex_intra_cluster_sync();

        for (int i = 0; i < TR; ++i)
        {
            //1. load Q_i, O_i, l_i, m_i
            if (flex_is_dm_core())
            {
                flex_dma_1d(local(0),hbm_south(pos.x,0), O_tilesize);
                flex_dma_1d(local(0),hbm_west(pos.y,0), Q_tilesize);
                flex_dma_1d(local(0),hbm_west(pos.y,0), m_tilesize);
                flex_dma_1d(local(0),hbm_west(pos.y,0), l_tilesize);
            }
            flex_intra_cluster_sync();

            if (flex_is_first_core())
            {
                //2. K transposition
                flex_mtxtran_set_M(D);
                flex_mtxtran_set_N(BC);
                flex_mtxtran_trigger_block();

                //3. Q*KT
                flex_redmule_set_M(0, BR);
                flex_redmule_set_N(0, D);
                flex_redmule_set_K(0, BC);
                flex_redmule_trigger_async(0);
                flex_redmule_trigger_wait(0);

                //4. rowmax(S_ij)
                flex_vecteng_set_M(BR);
                flex_vecteng_set_N(BC);
                flex_vecteng_trigger_RowMax();

                //5. exp(S_ij - m_ij)
                flex_vecteng_set_M(BR);
                flex_vecteng_set_N(BC);
                flex_vecteng_trigger_ExpSubMtx();

                //6. rowsum(P_ij)
                flex_vecteng_set_M(BR);
                flex_vecteng_set_N(BC);
                flex_vecteng_trigger_RowSum();

                //7. max(m_i, m_ij) 
                flex_vecteng_set_M(BR);
                flex_vecteng_trigger_VMax();

                //8. exp(m_i - m_i-new).l_i + exp(m_ij - m_i-new).l_ij
                flex_vecteng_set_M(BR);
                flex_vecteng_trigger_ExpSubV();
                flex_vecteng_trigger_VDotpV();
                flex_vecteng_trigger_ExpSubV();
                flex_vecteng_trigger_VDotpV();
                flex_vecteng_trigger_VAdd();

                //9. exp(m_ij - m_i-new) . (P_ij * V)
                flex_redmule_set_M(0, BR);
                flex_redmule_set_N(0, BC);
                flex_redmule_set_K(0, D);
                flex_redmule_trigger_async(0);
                flex_redmule_trigger_wait(0);
                flex_vecteng_set_M(BR);
                flex_vecteng_set_N(D);
                flex_vecteng_trigger_RowDotpV();

                //10. l_i . exp(m_i - m_i-new) . O_i
                flex_vecteng_set_M(BR);
                flex_vecteng_set_N(D);
                flex_vecteng_trigger_RowDotpV();
                flex_vecteng_trigger_RowDotpV();

                //11. final O
                flex_vecteng_set_M(BR);
                flex_vecteng_set_N(D);
                flex_vecteng_trigger_MtxAdd();
                flex_vecteng_trigger_RowDivV();
            }
            flex_intra_cluster_sync();

            //12. Store back O, m_i, l_i
            if (flex_is_dm_core())
            {
                flex_dma_1d(hbm_south(pos.x,0), local(0), O_tilesize);
                flex_dma_1d(hbm_west(pos.y,0), local(0), m_tilesize);
                flex_dma_1d(hbm_west(pos.y,0), local(0), l_tilesize);
            }
            flex_intra_cluster_sync();
        }
    }
}


void llm_mha_flash_attention(uint32_t sequence_length, uint32_t embedding_length, uint32_t head_dimension, uint32_t num_head, uint32_t batch_size, uint32_t elem_size){
    uint32_t BR = head_dimension;
    uint32_t BC = ARCH_CLUSTER_TCDM_SIZE/(4 * elem_size * head_dimension);
    uint32_t TR = sequence_length/BR;
    uint32_t TC = sequence_length/BC;
    uint32_t D = head_dimension;
    uint32_t cid = flex_get_cluster_id();
    uint32_t num_head_left = num_head * batch_size;

    if (flex_is_first_core() && cid == 0)
    {
        flex_log(TR * TC);
    }

    flex_global_barrier_xy();

    while(num_head_left > 0){
        if (cid < num_head_left)
        {
            llm_mha_flash_attention_each_head_orignal_in_paper(BR, BC, TR, TC, D, elem_size);
        } 
        num_head_left = (ARCH_NUM_CLUSTER_X*ARCH_NUM_CLUSTER_Y > num_head_left)? 0: (num_head_left - ARCH_NUM_CLUSTER_X*ARCH_NUM_CLUSTER_Y);
        flex_global_barrier_xy();
    }
}

#endif