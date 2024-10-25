#ifndef _LLM_MHA_FLATTEN_ATTENTION_H_
#define _LLM_MHA_FLATTEN_ATTENTION_H_

#include <math.h>
#include "snrt.h"
#include "flex_runtime.h"
#include "flex_vecteng.h"
#include "flex_mtxtran.h"
#include "flex_redmule.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"

void baseline_rowwise_broadcast(FlexPosition pos, uint32_t length, uint32_t elem_size){
    for (int i = 1; i < ARCH_NUM_CLUSTER_X; ++i)
    {
        if (flex_is_dm_core() && pos.x == i)
        {
            flex_dma_1d(local(0),remote_pos(left_pos(pos),0), length * elem_size);
        }
        flex_global_barrier_xy();
    }
}

void baseline_colwise_broadcast(FlexPosition pos, uint32_t length, uint32_t elem_size){
    for (int i = 1; i < ARCH_NUM_CLUSTER_Y; ++i)
    {
        if (flex_is_dm_core() && pos.y == i)
        {
            flex_dma_1d(local(0),remote_pos(bottom_pos(pos),0), length * elem_size);
        }
        flex_global_barrier_xy();
    }
}

void baseline_rowwise_reduction_sum(FlexPosition pos, uint32_t length, uint32_t elem_size){
    for (int i = ARCH_NUM_CLUSTER_X-2; i >= 0; --i)
    {
        if (pos.x == i)
        {
            if (flex_is_dm_core())
            {
                flex_dma_1d(local(0),remote_pos(right_pos(pos),0), length * elem_size);
            }
            flex_intra_cluster_sync();
            if (flex_is_first_core()){
                flex_vecteng_set_M(length);
                flex_vecteng_trigger_VAdd();
            }
        }
        flex_global_barrier_xy();
    }
}

void llm_mha_flatten_attention_each_head_double_buffer_no_collective(uint32_t T, uint32_t D, uint32_t elem_size){
    FlexPosition pos = get_pos(flex_get_cluster_id());
    uint32_t Q_tilesize = T*D*elem_size;
    uint32_t K_tilesize = T*D*elem_size;
    uint32_t V_tilesize = T*D*elem_size;
    uint32_t O_tilesize = T*D*elem_size;
    uint32_t m_tilesize = T*elem_size;
    uint32_t l_tilesize = T*elem_size;

    //2. Broadcast Q row-wise
    baseline_rowwise_broadcast(pos, T*D, elem_size);

    //3. Broadcast K col-wise
    baseline_colwise_broadcast(pos, T*D, elem_size);

    if (flex_is_first_core())
    {
        //4. K transposition
        flex_mtxtran_set_M(D);
        flex_mtxtran_set_N(T);
        flex_mtxtran_trigger_block();

        //5. Q*KT
        flex_redmule_set_M(0, T);
        flex_redmule_set_N(0, D);
        flex_redmule_set_K(0, T);
        flex_redmule_trigger_async(0);
        flex_redmule_trigger_wait(0);

        //6. Local Max
        flex_vecteng_set_M(T);
        flex_vecteng_set_N(T);
        flex_vecteng_trigger_RowMax();
    }
    if (flex_is_dm_core() && pos.x == 0)
    {
        //DoubleBuffering: Load Next V from west HBM
        flex_dma_1d(local(0),hbm_west(pos.y,0), V_tilesize);
        //DoubleBuffering: Store back previous output
        flex_dma_1d(hbm_west(pos.y,0), local(0), O_tilesize);
    }
    flex_global_barrier_xy();

    //7. Global Max
    baseline_rowwise_broadcast(pos, T, elem_size);
    baseline_rowwise_reduction_sum(pos, T, elem_size);

    //8. Local l_i
    if (flex_is_first_core())
    {
        //6. exp(S_ij - m_ij)
        flex_vecteng_set_M(T);
        flex_vecteng_set_N(T);
        flex_vecteng_trigger_ExpSubMtx();

        //7. rowsum(P_ij)
        flex_vecteng_set_M(T);
        flex_vecteng_set_N(T);
        flex_vecteng_trigger_RowSum();
    }
    flex_global_barrier_xy();

    //9. Global l_i
    baseline_rowwise_broadcast(pos, T, elem_size);
    baseline_rowwise_reduction_sum(pos, T, elem_size);

    //10. Final softmax
    if (flex_is_first_core())
    {
        //9. P_ij / l_i
        flex_vecteng_set_M(T);
        flex_vecteng_set_N(T);
        flex_vecteng_trigger_RowDivV();
    }
    flex_global_barrier_xy();

    //12. Broadcast V row-wise
    baseline_rowwise_broadcast(pos, T*D, elem_size);

    //13. SV Matmul
    if (flex_is_first_core())
    {
        flex_redmule_set_M(0, T);
        flex_redmule_set_N(0, T);
        flex_redmule_set_K(0, D);
        flex_redmule_trigger_async(0);
        flex_redmule_trigger_wait(0);
    }
    if (flex_is_dm_core())
    {
        //DoubleBuffering: Load Next Q and K to edge cluster
        if (pos.x == 0 || pos.y == 0)
        {
            if(pos.x == 0){
                /* clusters at west edge hbm transfer*/
                flex_dma_async_1d(local(0),hbm_west(pos.y,0), Q_tilesize);
            }

            if (pos.y == 0)
            {
                /* clusters at south edge hbm transfer*/
                flex_dma_async_1d(local(0),hbm_south(pos.x,0), K_tilesize);
            }
            flex_dma_async_wait_all();
        }
    }
    flex_global_barrier_xy();

    //14. Reduction O row-wise
    baseline_rowwise_reduction_sum(pos, T*D, elem_size);

}

void llm_mha_flatten_attention_each_head_double_buffer(uint32_t T, uint32_t D, uint32_t elem_size){
    FlexPosition pos = get_pos(flex_get_cluster_id());
    uint32_t Q_tilesize = T*D*elem_size;
    uint32_t K_tilesize = T*D*elem_size;
    uint32_t V_tilesize = T*D*elem_size;
    uint32_t O_tilesize = T*D*elem_size;
    uint32_t m_tilesize = T*elem_size;
    uint32_t l_tilesize = T*elem_size;
    uint32_t pseudo_noc_broadcast_tail = (ARCH_NOC_LINK_WIDTH/(elem_size * 8)) * (ARCH_NUM_CLUSTER_X - 1);

    //2. Broadcast Q row-wise
    if (flex_is_dm_core() && pos.x != 0)
    {
        flex_dma_1d(local(0),remote_pos(left_pos(pos),0), Q_tilesize + pseudo_noc_broadcast_tail);
    }
    flex_global_barrier_xy();

    //3. Broadcast K col-wise
    if (flex_is_dm_core() && pos.y != 0)
    {
        flex_dma_1d(local(0),remote_pos(bottom_pos(pos),0), K_tilesize + pseudo_noc_broadcast_tail);
    }
    flex_global_barrier_xy();

    if (flex_is_first_core())
    {
        //4. K transposition
        flex_mtxtran_set_M(D);
        flex_mtxtran_set_N(T);
        flex_mtxtran_trigger_block();

        //5. Q*KT
        flex_redmule_set_M(0, T);
        flex_redmule_set_N(0, D);
        flex_redmule_set_K(0, T);
        flex_redmule_trigger_async(0);
        flex_redmule_trigger_wait(0);

        //6. Local Max
        flex_vecteng_set_M(T);
        flex_vecteng_set_N(T);
        flex_vecteng_trigger_RowMax();
    } else if (flex_is_dm_core() && (pos.x == 0))
    {
        //DoubleBuffering: Load Next V from west HBM
        flex_dma_1d(local(0),hbm_west(pos.y,0), V_tilesize);
        //DoubleBuffering: Store back previous output
        flex_dma_1d(hbm_west(pos.y,0), local(0), O_tilesize);
    }
    flex_global_barrier_xy();

    //7. Global Max
    if (flex_is_dm_core() && pos.x != 0)
    {
        flex_dma_1d(local(0),remote_pos(left_pos(pos),0), m_tilesize + pseudo_noc_broadcast_tail);
    }
    flex_global_barrier_xy();
    if (flex_is_dm_core() && pos.x != (ARCH_NUM_CLUSTER_X-1))
    {
        flex_dma_1d(local(0),remote_pos(right_pos(pos),0), m_tilesize + pseudo_noc_broadcast_tail);
    }
    flex_global_barrier_xy();

    //8. Local l_i
    if (flex_is_first_core())
    {
        //6. exp(S_ij - m_ij)
        flex_vecteng_set_M(T);
        flex_vecteng_set_N(T);
        flex_vecteng_trigger_ExpSubMtx();

        //7. rowsum(P_ij)
        flex_vecteng_set_M(T);
        flex_vecteng_set_N(T);
        flex_vecteng_trigger_RowSum();
    }
    flex_global_barrier_xy();

    //9. Global l_i
    if (flex_is_dm_core() && pos.x != 0)
    {
        flex_dma_1d(local(0),remote_pos(left_pos(pos),0), l_tilesize + pseudo_noc_broadcast_tail);
    }
    flex_global_barrier_xy();
    if (flex_is_dm_core() && pos.x != (ARCH_NUM_CLUSTER_X-1))
    {
        flex_dma_1d(local(0),remote_pos(right_pos(pos),0), l_tilesize + pseudo_noc_broadcast_tail);
    }
    flex_global_barrier_xy();

    //10. Final softmax
    if (flex_is_first_core())
    {
        //9. P_ij / l_i
        flex_vecteng_set_M(T);
        flex_vecteng_set_N(T);
        flex_vecteng_trigger_RowDivV();
    }
    flex_global_barrier_xy();

    //12. Broadcast V row-wise
    if (flex_is_dm_core() && pos.x != 0)
    {
        flex_dma_1d(local(0),remote_pos(left_pos(pos),0), V_tilesize + pseudo_noc_broadcast_tail);
    }
    flex_global_barrier_xy();

    //13. SV Matmul
    if (flex_is_first_core())
    {
        flex_redmule_set_M(0, T);
        flex_redmule_set_N(0, T);
        flex_redmule_set_K(0, D);
        flex_redmule_trigger_async(0);
        flex_redmule_trigger_wait(0);
    }
    if (flex_is_dm_core())
    {
        //DoubleBuffering: Load Next Q and K to edge cluster
        if (pos.x == 0 || pos.y == 0)
        {
            if(pos.x == 0){
                /* clusters at west edge hbm transfer*/
                flex_dma_async_1d(local(0),hbm_west(pos.y,0), Q_tilesize);
            }

            if (pos.y == 0)
            {
                /* clusters at south edge hbm transfer*/
                flex_dma_async_1d(local(0),hbm_south(pos.x,0), K_tilesize);
            }
            flex_dma_async_wait_all();
        }
    }
    flex_global_barrier_xy();

    //14. Reduction O row-wise
    if (flex_is_dm_core() && pos.x != (ARCH_NUM_CLUSTER_X-1))
    {
        flex_dma_1d(local(0),remote_pos(right_pos(pos),0), O_tilesize + pseudo_noc_broadcast_tail);
    }
    flex_global_barrier_xy();

}



void llm_mha_flatten_attention_each_head(uint32_t T, uint32_t D, uint32_t elem_size){
    FlexPosition pos = get_pos(flex_get_cluster_id());
    uint32_t Q_tilesize = T*D*elem_size;
    uint32_t K_tilesize = T*D*elem_size;
    uint32_t V_tilesize = T*D*elem_size;
    uint32_t O_tilesize = T*D*elem_size;
    uint32_t m_tilesize = T*elem_size;
    uint32_t l_tilesize = T*elem_size;
    uint32_t pseudo_noc_broadcast_tail = (ARCH_NOC_LINK_WIDTH/(elem_size * 8)) * (ARCH_NUM_CLUSTER_X - 1);

    //1. Load Q and K to edge cluster
    if (pos.x == 0 || pos.y == 0)
    {
        if (flex_is_dm_core())
        {
            if(pos.x == 0){
                /* clusters at west edge hbm transfer*/
                flex_dma_async_1d(local(0),hbm_west(pos.y,0), Q_tilesize);
            }

            if (pos.y == 0)
            {
                /* clusters at south edge hbm transfer*/
                flex_dma_async_1d(local(0),hbm_south(pos.x,0), K_tilesize);
            }

            flex_dma_async_wait_all();
        }
    }
    flex_global_barrier_xy();

    //2. Broadcast Q row-wise
    if (flex_is_dm_core() && pos.x != 0)
    {
        flex_dma_1d(local(0),remote_pos(left_pos(pos),0), Q_tilesize + pseudo_noc_broadcast_tail);
    }
    flex_global_barrier_xy();

    //3. Broadcast K col-wise
    if (flex_is_dm_core() && pos.y != 0)
    {
        flex_dma_1d(local(0),remote_pos(bottom_pos(pos),0), K_tilesize + pseudo_noc_broadcast_tail);
    }
    flex_global_barrier_xy();

    if (flex_is_first_core())
    {
        //4. K transposition
        flex_mtxtran_set_M(D);
        flex_mtxtran_set_N(T);
        flex_mtxtran_trigger_block();

        //5. Q*KT
        flex_redmule_set_M(0, T);
        flex_redmule_set_N(0, D);
        flex_redmule_set_K(0, T);
        flex_redmule_trigger_async(0);
        flex_redmule_trigger_wait(0);

        //6. Local Max
        flex_vecteng_set_M(T);
        flex_vecteng_set_N(T);
        flex_vecteng_trigger_RowMax();
    }
    flex_global_barrier_xy();

    //7. Global Max
    if (flex_is_dm_core() && pos.x != 0)
    {
        flex_dma_1d(local(0),remote_pos(left_pos(pos),0), m_tilesize + pseudo_noc_broadcast_tail);
    }
    flex_global_barrier_xy();
    if (flex_is_dm_core() && pos.x != (ARCH_NUM_CLUSTER_X-1))
    {
        flex_dma_1d(local(0),remote_pos(right_pos(pos),0), m_tilesize + pseudo_noc_broadcast_tail);
    }
    flex_global_barrier_xy();

    //8. Local l_i
    if (flex_is_first_core())
    {
        //6. exp(S_ij - m_ij)
        flex_vecteng_set_M(T);
        flex_vecteng_set_N(T);
        flex_vecteng_trigger_ExpSubMtx();

        //7. rowsum(P_ij)
        flex_vecteng_set_M(T);
        flex_vecteng_set_N(T);
        flex_vecteng_trigger_RowSum();
    }
    flex_global_barrier_xy();

    //9. Global l_i
    if (flex_is_dm_core() && pos.x != 0)
    {
        flex_dma_1d(local(0),remote_pos(left_pos(pos),0), l_tilesize + pseudo_noc_broadcast_tail);
    }
    flex_global_barrier_xy();
    if (flex_is_dm_core() && pos.x != (ARCH_NUM_CLUSTER_X-1))
    {
        flex_dma_1d(local(0),remote_pos(right_pos(pos),0), l_tilesize + pseudo_noc_broadcast_tail);
    }
    flex_global_barrier_xy();

    //10. Final softmax
    if (flex_is_first_core())
    {
        //9. P_ij / l_i
        flex_vecteng_set_M(T);
        flex_vecteng_set_N(T);
        flex_vecteng_trigger_RowDivV();
    }
    flex_global_barrier_xy();

    //11. Load V from west HBM
    if (flex_is_dm_core() && pos.x == 0)
    {
        flex_dma_1d(local(0),hbm_west(pos.y,0), V_tilesize);
    }
    flex_global_barrier_xy();

    //12. Broadcast V row-wise
    if (flex_is_dm_core() && pos.x != 0)
    {
        flex_dma_1d(local(0),remote_pos(left_pos(pos),0), V_tilesize + pseudo_noc_broadcast_tail);
    }
    flex_global_barrier_xy();

    //13. SV Matmul
    if (flex_is_first_core())
    {
        flex_redmule_set_M(0, T);
        flex_redmule_set_N(0, T);
        flex_redmule_set_K(0, D);
        flex_redmule_trigger_async(0);
        flex_redmule_trigger_wait(0);
    }
    flex_global_barrier_xy();

    //14. Reduction O row-wise
    if (flex_is_dm_core() && pos.x != (ARCH_NUM_CLUSTER_X-1))
    {
        flex_dma_1d(local(0),remote_pos(right_pos(pos),0), O_tilesize + pseudo_noc_broadcast_tail);
    }
    flex_global_barrier_xy();

    //15. Store back O
    if (flex_is_dm_core() && pos.x == 0)
    {
        flex_dma_1d(hbm_west(pos.y,0), local(0), O_tilesize);
    }
    flex_global_barrier_xy();
}

void llm_mha_flatten_attention(uint32_t sequence_length, uint32_t embedding_length, uint32_t head_dimension, uint32_t num_head, uint32_t batch_size, uint32_t elem_size){
    uint32_t T = sequence_length/ARCH_NUM_CLUSTER_X;
    uint32_t D = head_dimension;

    uint32_t cid = flex_get_cluster_id();

    if (flex_is_first_core() && cid == 0)
    {
        flex_log(num_head * batch_size);
    }

    flex_global_barrier_xy();

    for (int i = 0; i < num_head * batch_size; ++i)
    {
        llm_mha_flatten_attention_each_head_double_buffer_no_collective(T,D,elem_size);
    }
}

#endif