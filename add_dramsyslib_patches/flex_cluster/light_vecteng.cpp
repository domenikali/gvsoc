/*
 * Copyright (C) 2024 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Author: Chi     Zhang , ETH Zurich (chizhang@iis.ee.ethz.ch)
 * Note:
 *      Here we ignore real transposition in order to accelerate simulation
 */

#include <vp/vp.hpp>
#include <vp/itf/io.hpp>
#include <vp/itf/wire.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <list>
#include <queue>

class LightVectEng : public vp::Component
{

public:
    LightVectEng(vp::ComponentConf &config);

// private:
    static vp::IoReqStatus req(vp::Block *__this, vp::IoReq *req);
    static void event_handler(vp::Block *__this, vp::ClockEvent *event);

    vp::Trace           trace;
    vp::IoSlave         input_itf;
    vp::IoMaster        tcdm_itf;
    vp::ClockEvent * 	vecteng_event;

    //vecteng fsm
    vp::IoReq *         vecteng_query;
    int64_t             timer_start;
    int64_t             total_runtime;
    int64_t             num_vect_jobs;
    int64_t             idx_vect_jobs;

    //vecteng configuration
    uint32_t            tcdm_bank_width;
    uint32_t            tcdm_bank_number;
    uint32_t            elem_size;
    uint32_t            bandwidth;
    uint32_t            alu_latency;
    uint32_t            exp_latency;
    uint32_t            red_latency;

    //vecteng registers
    uint32_t            m_size;
    uint32_t            n_size;
    uint32_t            x_addr;
    uint32_t            y_addr;
};


extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new LightVectEng(config);
}

LightVectEng::LightVectEng(vp::ComponentConf &config)
    : vp::Component(config)
{
    //Initialize interface
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->input_itf.set_req_meth(&LightVectEng::req);
    this->new_slave_port("input", &this->input_itf);
    this->new_master_port("tcdm", &this->tcdm_itf);
    this->vecteng_event = this->event_new(&LightVectEng::event_handler);
    
    //Initialize configuration
    this->tcdm_bank_width   = get_js_config()->get("tcdm_bank_width")->get_int();
    this->tcdm_bank_number  = get_js_config()->get("tcdm_bank_number")->get_int();
    this->elem_size         = get_js_config()->get("elem_size")->get_int();
    this->bandwidth         = this->tcdm_bank_width * this->tcdm_bank_number;
    this->alu_latency       = get_js_config()->get("alu_latency")->get_int();
    this->exp_latency       = get_js_config()->get("exp_latency")->get_int();
    this->red_latency       = get_js_config()->get("red_latency")->get_int();

    //Initialize registers
    this->m_size            = 4;
    this->n_size            = 4;
    this->x_addr            = 0;
    this->y_addr            = 0;

    //Initialize FSM
    this->vecteng_query     = NULL;
    this->timer_start       = 0;
    this->total_runtime     = 0;
    this->num_vect_jobs     = 0;
    this->idx_vect_jobs     = 0;
}


vp::IoReqStatus LightVectEng::req(vp::Block *__this, vp::IoReq *req)
{
    LightVectEng *_this = (LightVectEng *)__this;

    uint64_t offset = req->get_addr();
    uint8_t *data = req->get_data();
    uint64_t size = req->get_size();
    bool is_write = req->get_is_write();

    _this->trace.msg(vp::Trace::LEVEL_TRACE,"[LightVectEng] access (offset: 0x%x, size: 0x%x, is_write: %d, data:%x)\n", offset, size, is_write, *(uint32_t *)data);

    if ((is_write == 0) && (_this->vecteng_query == NULL))
    {
        /************************
        *  Synchronize Trigger  *
        ************************/
        //Sanity Check
        _this->trace.msg(vp::Trace::LEVEL_TRACE,"[LightVectEng] vecteng configuration (M-N): %d, %d\n", _this->m_size, _this->n_size);
        if ((_this->m_size == 0)||(_this->n_size == 0))
        {
            _this->trace.fatal("[LightVectEng] INVALID vecteng configuration (M-N): %d, %d\n", _this->m_size, _this->n_size);
            return vp::IO_REQ_OK;
        }

        uint32_t vecteng_runtime = 0;
        //Calculation on Runtime
        //Check Which Operation is going to take place
        if (offset == 0)
        {
            //max(x)
            uint32_t num_load_per_row   = (_this->n_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            uint32_t num_store          = (_this->m_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            vecteng_runtime = num_load_per_row * _this->m_size + num_store + _this->red_latency;

            _this->idx_vect_jobs = 0;
        } else
        if (offset == 4)
        {
            //sum(x)
            uint32_t num_load_per_row   = (_this->n_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            uint32_t num_store          = (_this->m_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            vecteng_runtime = num_load_per_row * _this->m_size + num_store + _this->red_latency;

            _this->idx_vect_jobs = 1;
        } else
        if (offset == 8)
        {
            //exp(x-max)
            uint32_t num_load_per_row   = (_this->n_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            uint32_t num_load_max       = (_this->m_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            uint32_t num_store          = (_this->m_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            vecteng_runtime    = num_load_max + num_load_per_row * _this->m_size + num_store + _this->alu_latency + _this->exp_latency;

            _this->idx_vect_jobs = 2;
        } else
        if (offset == 12)
        {
            //mtx/s
            uint32_t num_load_per_row   = (_this->n_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            uint32_t num_load_s         = (_this->m_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            uint32_t num_store_per_row  = (_this->n_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            vecteng_runtime    = num_load_s + (num_load_per_row + num_store_per_row) * _this->m_size + _this->alu_latency;

            _this->idx_vect_jobs = 3;
        } else
        if (offset == 16)
        {
            //mtx dotp v
            uint32_t num_load_per_row   = (_this->n_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            uint32_t num_load_v         = (_this->m_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            uint32_t num_store_per_row  = (_this->n_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            vecteng_runtime    = num_load_v + (num_load_per_row + num_store_per_row) * _this->m_size + _this->alu_latency;

            _this->idx_vect_jobs = 4;
        } else
        if (offset == 20)
        {
            //mtx add mtx
            uint32_t num_load_per_row   = (_this->n_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            vecteng_runtime    = (num_load_per_row * 3) * _this->m_size + _this->alu_latency;

            _this->idx_vect_jobs = 5;
        } else
        if (offset == 24)
        {
            //v dotp v
            uint32_t num_load_v         = (_this->m_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            vecteng_runtime    = 3 * num_load_v + _this->alu_latency;

            _this->idx_vect_jobs = 6;
        } else
        if (offset == 28)
        {
            //v add v
            uint32_t num_load_v         = (_this->m_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            vecteng_runtime    = 3 * num_load_v + _this->alu_latency;

            _this->idx_vect_jobs = 7;
        } else
        if (offset == 32)
        {
            //v max v
            uint32_t num_load_v         = (_this->m_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            vecteng_runtime    = 3 * num_load_v + _this->alu_latency;

            _this->idx_vect_jobs = 8;
        } else
        if (offset == 36)
        {
            //exp(v-v)
            uint32_t num_load_v         = (_this->m_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
            vecteng_runtime    = 3 * num_load_v + _this->alu_latency + _this->exp_latency;

            _this->idx_vect_jobs = 9;
        } else {
            _this->idx_vect_jobs = 10;
        }
        _this->timer_start = _this->time.get_time();

        //Save Query
        _this->vecteng_query = req;

        //Trigger timer
        _this->event_enqueue(_this->vecteng_event, vecteng_runtime);

        return vp::IO_REQ_PENDING;

    } else {
        uint32_t value = *(uint32_t *)data;

        switch (offset) {
            case 0:
                _this->m_size = value;
                _this->trace.msg(vp::Trace::LEVEL_TRACE,"[LightVectEng] Set M size 0x%x\n", value);
                break;
            case 4:
                _this->n_size = value;
                _this->trace.msg(vp::Trace::LEVEL_TRACE,"[LightVectEng] Set N size 0x%x\n", value);
                break;
            case 8:
                _this->x_addr = value;
                _this->trace.msg(vp::Trace::LEVEL_TRACE,"[LightVectEng] Set X addr 0x%x\n", value);
                break;
            case 12:
                _this->y_addr = value;
                _this->trace.msg(vp::Trace::LEVEL_TRACE,"[LightVectEng] Set Y addr 0x%x\n", value);
                break;
            default:
                _this->trace.msg("[LightVectEng] write to INVALID address\n");
        }
    }

    return vp::IO_REQ_OK;
}


void LightVectEng::event_handler(vp::Block *__this, vp::ClockEvent *event) {
    LightVectEng *_this = (LightVectEng *)__this;
    int64_t start_time_ns   = (_this->timer_start)/1000;
    int64_t end_time_ns     = (_this->time.get_time())/1000;
    int64_t period_ns       = end_time_ns - start_time_ns;
    _this->total_runtime   += period_ns;
	_this->num_vect_jobs   += 1;
    _this->vecteng_query->get_resp_port()->resp(_this->vecteng_query);
    _this->vecteng_query = NULL;
    switch (_this->idx_vect_jobs) {
            case 0:     _this->trace.msg("[LightVectEng] Job: max(x)\n"); break;
            case 1:     _this->trace.msg("[LightVectEng] Job: sum(x)\n"); break;
            case 2:     _this->trace.msg("[LightVectEng] Job: exp(x-max)\n"); break;
            case 3:     _this->trace.msg("[LightVectEng] Job: mtx/s\n"); break;
            case 4:     _this->trace.msg("[LightVectEng] Job: mtx.v\n"); break;
            case 5:     _this->trace.msg("[LightVectEng] Job: mtx+mtx\n"); break;
            case 6:     _this->trace.msg("[LightVectEng] Job: v.v\n"); break;
            case 7:     _this->trace.msg("[LightVectEng] Job: v+v\n"); break;
            case 8:     _this->trace.msg("[LightVectEng] Job: max(v,v)\n"); break;
            case 9:     _this->trace.msg("[LightVectEng] Job: exp(v-v)\n"); break;
            default:    _this->trace.msg("[LightVectEng] Invalid Job\n");
    }
    _this->trace.msg("[LightVectEng] Finished : %0d ns ---> %0d ns | period = %0d ns | runtime = %0d ns | id = %0d\n", start_time_ns, end_time_ns, period_ns, _this->total_runtime, _this->num_vect_jobs);
}
