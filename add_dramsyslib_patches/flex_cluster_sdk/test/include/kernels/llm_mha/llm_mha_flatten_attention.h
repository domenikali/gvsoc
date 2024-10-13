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

void llm_mha_flatten_attention_each_head_naive(uint32_t T, uint32_t D, uint32_t elem_size){
	FlexPosition pos = get_pos(flex_get_cluster_id());
	uint32_t QK_tilesize = T*D*elem_size;
	uint32_t vector_size = T*elem_size;

   	//1. Load Q and K to edge cluster
   	if (pos.x == 0 || pos.y == 0)
   	{
   		if (flex_is_dm_core())
   		{
   			if(pos.x == 0){
                /* clusters at west edge hbm transfer*/
                flex_dma_async_1d(local(0),hbm_west(pos.y,0), QK_tilesize);
            }

            if (pos.y == 0)
            {
                /* clusters at south edge hbm transfer*/
                flex_dma_async_1d(local(QK_tilesize),hbm_south(0,0), QK_tilesize);
            }

            flex_dma_async_wait_all();
   		}
   	}
   	flex_global_barrier_xy();

   	//2. Broadcast row-wise
   	for (int i = 0; i < ARCH_NUM_CLUSTER_X-1; ++i)
   	{
   		if (flex_is_dm_core() && pos.x == (i+1))
	   	{
	   		flex_dma_async_1d(local(0),remote_pos(left_pos(pos),0), QK_tilesize);
	   		flex_dma_async_wait_all();
	   	}
	   	flex_global_barrier_xy();
   	}

   	//3. Broadcast col-wise
   	for (int i = 0; i < ARCH_NUM_CLUSTER_Y-1; ++i)
   	{
   		if (flex_is_dm_core() && pos.y == (i+1))
	   	{
	   		flex_dma_async_1d(local(QK_tilesize),remote_pos(bottom_pos(pos),QK_tilesize), QK_tilesize);
	   		flex_dma_async_wait_all();
	   	}
	   	flex_global_barrier_xy();
   	}

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
        flex_vecteng_trigger_Max();
   	}
   	flex_global_barrier_xy();

   	//7. Global Max
   	for (int i = 0; i < ARCH_NUM_CLUSTER_X-1; ++i)
   	{
   		if (flex_is_dm_core() && pos.x == (i+1))
	   	{
	   		flex_dma_async_1d(local(0),remote_pos(left_pos(pos),0), vector_size);
	   		flex_dma_async_wait_all();
	   	}
	   	flex_global_barrier_xy();
   	}
	for (int i = 0; i < ARCH_NUM_CLUSTER_X-1; ++i)
	{
		if (flex_is_dm_core() && pos.x == (ARCH_NUM_CLUSTER_X-2-i))
	   	{
	   		flex_dma_async_1d(local(0),remote_pos(right_pos(pos),0), vector_size);
	   		flex_dma_async_wait_all();
	   	}
	   	flex_global_barrier_xy();
	}


   	//8. Local SumExpSub
   	if (flex_is_first_core())
   	{
   		flex_vecteng_set_M(T);
        flex_vecteng_set_N(T);
        flex_vecteng_trigger_SumExpSub();
   	}
   	flex_global_barrier_xy();

   	//9. Global SumExpSub
   	for (int i = 0; i < ARCH_NUM_CLUSTER_X-1; ++i)
   	{
   		if (flex_is_dm_core() && pos.x == (i+1))
	   	{
	   		flex_dma_async_1d(local(0),remote_pos(left_pos(pos),0), vector_size);
	   		flex_dma_async_wait_all();
	   	}
	   	flex_global_barrier_xy();
   	}
	for (int i = 0; i < ARCH_NUM_CLUSTER_X-1; ++i)
	{
		if (flex_is_dm_core() && pos.x == (ARCH_NUM_CLUSTER_X-2-i))
	   	{
	   		flex_dma_async_1d(local(0),remote_pos(right_pos(pos),0), vector_size);
	   		flex_dma_async_wait_all();
	   	}
	   	flex_global_barrier_xy();
	}

   	//10. Final softmax
   	if (flex_is_first_core())
   	{
   		flex_vecteng_set_M(T);
        flex_vecteng_set_N(T);
        flex_vecteng_trigger_ExpMaxDiv();
   	}
   	flex_global_barrier_xy();

   	//11. Load V from west HBM
   	if (flex_is_dm_core() && pos.x == 0)
   	{
   		flex_dma_async_1d(local(0),hbm_west(pos.y,0), QK_tilesize);
   		flex_dma_async_wait_all();
   	}
   	flex_global_barrier_xy();

   	//12. Broadcast row-wise
   	if (flex_is_dm_core() && pos.x != 0)
   	{
   		flex_dma_async_1d(local(0),remote_pos(left_pos(pos),0), QK_tilesize);
   		flex_dma_async_wait_all();
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

   	//14. Reduction row-wise
   	for (int i = 0; i < ARCH_NUM_CLUSTER_X-1; ++i)
	{
		if (flex_is_dm_core() && pos.x == (ARCH_NUM_CLUSTER_X-2-i))
	   	{
	   		flex_dma_async_1d(local(0),remote_pos(right_pos(pos),0), QK_tilesize);
	   		flex_dma_async_wait_all();
	   	}
	   	flex_global_barrier_xy();
	}

   	//15. Store back output
   	if (flex_is_dm_core() && pos.x == 0)
   	{
   		flex_dma_async_1d(hbm_west(pos.y,0), local(0), QK_tilesize);
   		flex_dma_async_wait_all();
   	}
   	flex_global_barrier_xy();
}

void llm_mha_flatten_attention_each_head(uint32_t T, uint32_t D, uint32_t elem_size){
	FlexPosition pos = get_pos(flex_get_cluster_id());
	uint32_t QK_tilesize = T*D*elem_size;
	uint32_t vector_size = T*elem_size;

   	//1. Load Q and K to edge cluster
   	if (pos.x == 0 || pos.y == 0)
   	{
   		if (flex_is_dm_core())
   		{
   			if(pos.x == 0){
                /* clusters at west edge hbm transfer*/
                flex_dma_async_1d(local(0),hbm_west(pos.y,0), QK_tilesize);
            }

            if (pos.y == 0)
            {
                /* clusters at south edge hbm transfer*/
                flex_dma_async_1d(local(QK_tilesize),hbm_south(0,0), QK_tilesize);
            }

            flex_dma_async_wait_all();
   		}
   	}
   	flex_global_barrier_xy();

   	//2. Broadcast row-wise
   	if (flex_is_dm_core() && pos.x != 0)
   	{
   		flex_dma_async_1d(local(0),remote_pos(left_pos(pos),0), QK_tilesize);
   		flex_dma_async_wait_all();
   	}
   	flex_global_barrier_xy();

   	//3. Broadcast col-wise
   	if (flex_is_dm_core() && pos.y != 0)
   	{
   		flex_dma_async_1d(local(QK_tilesize),remote_pos(bottom_pos(pos),QK_tilesize), QK_tilesize);
   		flex_dma_async_wait_all();
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
        flex_vecteng_trigger_Max();
   	}
   	flex_global_barrier_xy();

   	//7. Global Max
   	if (flex_is_dm_core() && pos.x != 0)
   	{
   		flex_dma_async_1d(local(0),remote_pos(left_pos(pos),0), vector_size);
   		flex_dma_async_wait_all();
   	}
   	flex_global_barrier_xy();
   	if (flex_is_dm_core() && pos.x != (ARCH_NUM_CLUSTER_X-1))
   	{
   		flex_dma_async_1d(local(0),remote_pos(right_pos(pos),0), vector_size);
   		flex_dma_async_wait_all();
   	}
   	flex_global_barrier_xy();

   	//8. Local SumExpSub
   	if (flex_is_first_core())
   	{
   		flex_vecteng_set_M(T);
        flex_vecteng_set_N(T);
        flex_vecteng_trigger_SumExpSub();
   	}
   	flex_global_barrier_xy();

   	//9. Global SumExpSub
   	if (flex_is_dm_core() && pos.x != 0)
   	{
   		flex_dma_async_1d(local(0),remote_pos(left_pos(pos),0), vector_size);
   		flex_dma_async_wait_all();
   	}
   	flex_global_barrier_xy();
   	if (flex_is_dm_core() && pos.x != (ARCH_NUM_CLUSTER_X-1))
   	{
   		flex_dma_async_1d(local(0),remote_pos(right_pos(pos),0), vector_size);
   		flex_dma_async_wait_all();
   	}
   	flex_global_barrier_xy();

   	//10. Final softmax
   	if (flex_is_first_core())
   	{
   		flex_vecteng_set_M(T);
        flex_vecteng_set_N(T);
        flex_vecteng_trigger_ExpMaxDiv();
   	}
   	flex_global_barrier_xy();

   	//11. Load V from west HBM
   	if (flex_is_dm_core() && pos.x == 0)
   	{
   		flex_dma_async_1d(local(0),hbm_west(pos.y,0), QK_tilesize);
   		flex_dma_async_wait_all();
   	}
   	flex_global_barrier_xy();

   	//12. Broadcast row-wise
   	if (flex_is_dm_core() && pos.x != 0)
   	{
   		flex_dma_async_1d(local(0),remote_pos(left_pos(pos),0), QK_tilesize);
   		flex_dma_async_wait_all();
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

   	//14. Reduction row-wise
   	if (flex_is_dm_core() && pos.x != (ARCH_NUM_CLUSTER_X-1))
   	{
   		flex_dma_async_1d(local(0),remote_pos(right_pos(pos),0), QK_tilesize);
   		flex_dma_async_wait_all();
   	}
   	flex_global_barrier_xy();

   	//15. Store back output
   	if (flex_is_dm_core() && pos.x == 0)
   	{
   		flex_dma_async_1d(hbm_west(pos.y,0), local(0), QK_tilesize);
   		flex_dma_async_wait_all();
   	}
   	flex_global_barrier_xy();
}

void llm_mha_flatten_attention(uint32_t sequence_length, uint32_t embedding_length, uint32_t head_dimension, uint32_t num_head, uint32_t elem_size){
	uint32_t T = sequence_length/ARCH_NUM_CLUSTER_X;
	uint32_t D = head_dimension;

	flex_global_barrier_xy();

    for (int i = 0; i < num_head; ++i)
    {
    	llm_mha_flatten_attention_each_head(T,D,elem_size);
    }
}

#endif