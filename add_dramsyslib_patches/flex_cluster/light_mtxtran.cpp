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

class LightMtxTran : public vp::Component
{

public:
    LightMtxTran(vp::ComponentConf &config);

// private:
    static vp::IoReqStatus req(vp::Block *__this, vp::IoReq *req);
    static void event_handler(vp::Block *__this, vp::ClockEvent *event);

    vp::Trace           trace;
    vp::IoSlave         input_itf;
    vp::IoMaster        tcdm_itf;
    vp::ClockEvent * 	mtxtran_event;

    //mtxtran fsm
    vp::IoReq *         mtxtran_query;
    int64_t             timer_start;
    int64_t             total_runtime;
    int64_t             num_transpose;

    //mtxtran configuration
    uint32_t            tcdm_bank_width;
    uint32_t            tcdm_bank_number;
    uint32_t            elem_size;
    uint32_t            bandwidth;

    //mtxtran registers
    uint32_t            m_size;
    uint32_t            n_size;
    uint32_t            x_addr;
    uint32_t            y_addr;
};


extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new LightMtxTran(config);
}

LightMtxTran::LightMtxTran(vp::ComponentConf &config)
    : vp::Component(config)
{
    //Initialize interface
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->input_itf.set_req_meth(&LightMtxTran::req);
    this->new_slave_port("input", &this->input_itf);
    this->new_master_port("tcdm", &this->tcdm_itf);
    this->mtxtran_event = this->event_new(&LightMtxTran::event_handler);
    
    
    //Initialize configuration
    this->tcdm_bank_width   = get_js_config()->get("tcdm_bank_width")->get_int();
    this->tcdm_bank_number  = get_js_config()->get("tcdm_bank_number")->get_int();
    this->elem_size         = get_js_config()->get("elem_size")->get_int();
    this->bandwidth         = this->tcdm_bank_width * this->tcdm_bank_number;

    //Initialize registers
    this->m_size            = 4;
    this->n_size            = 4;
    this->x_addr            = 0;
    this->y_addr            = 0;

    //Initialize FSM
    this->mtxtran_query     = NULL;
    this->timer_start       = 0;
    this->total_runtime     = 0;
    this->num_transpose     = 0;
}


vp::IoReqStatus LightMtxTran::req(vp::Block *__this, vp::IoReq *req)
{
    LightMtxTran *_this = (LightMtxTran *)__this;

    uint64_t offset = req->get_addr();
    uint8_t *data = req->get_data();
    uint64_t size = req->get_size();
    bool is_write = req->get_is_write();

    _this->trace.msg(vp::Trace::LEVEL_TRACE,"[LightMtxTran] access (offset: 0x%x, size: 0x%x, is_write: %d, data:%x)\n", offset, size, is_write, *(uint32_t *)data);

    if ((is_write == 0) && (_this->mtxtran_query == NULL))
    {
        /************************
        *  Synchronize Trigger  *
        ************************/
        //Sanity Check
        _this->trace.msg(vp::Trace::LEVEL_TRACE,"[LightMtxTran] mtxtran configuration (M-N): %d, %d\n", _this->m_size, _this->n_size);
        if ((_this->m_size == 0)||(_this->n_size == 0))
        {
            _this->trace.fatal("[LightMtxTran] INVALID mtxtran configuration (M-N): %d, %d\n", _this->m_size, _this->n_size);
            return vp::IO_REQ_OK;
        }

        //Calculation on Runtime
        uint32_t num_transpose_tile_M 	= (_this->m_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
        uint32_t num_transpose_tile_N 	= (_this->n_size * _this->elem_size + _this->bandwidth - 1) / _this->bandwidth;
        uint32_t runtime_per_tile 		= 2 * _this->bandwidth / _this->elem_size;
        uint32_t transpose_runtime		= num_transpose_tile_M * num_transpose_tile_N * runtime_per_tile;
        _this->timer_start      		= _this->time.get_time();

        //Save Query
        _this->mtxtran_query = req;

        //Trigger timer
        _this->event_enqueue(_this->mtxtran_event, transpose_runtime);

        return vp::IO_REQ_PENDING;

    } else {
        uint32_t value = *(uint32_t *)data;

        switch (offset) {
            case 0:
                _this->m_size = value;
                _this->trace.msg(vp::Trace::LEVEL_TRACE,"[LightMtxTran] Set M size 0x%x\n", value);
                break;
            case 4:
                _this->n_size = value;
                _this->trace.msg(vp::Trace::LEVEL_TRACE,"[LightMtxTran] Set N size 0x%x\n", value);
                break;
            case 8:
                _this->x_addr = value;
                _this->trace.msg(vp::Trace::LEVEL_TRACE,"[LightMtxTran] Set X addr 0x%x\n", value);
                break;
            case 12:
                _this->y_addr = value;
                _this->trace.msg(vp::Trace::LEVEL_TRACE,"[LightMtxTran] Set Y addr 0x%x\n", value);
                break;
            default:
                _this->trace.msg("[LightMtxTran] write to INVALID address\n");
        }
    }

    return vp::IO_REQ_OK;
}


void LightMtxTran::event_handler(vp::Block *__this, vp::ClockEvent *event) {
    LightMtxTran *_this = (LightMtxTran *)__this;
    int64_t start_time_ns   = (_this->timer_start)/1000;
    int64_t end_time_ns     = (_this->time.get_time())/1000;
    int64_t period_ns       = end_time_ns - start_time_ns;
    _this->total_runtime   += period_ns;
	_this->num_transpose   += 1;
    _this->mtxtran_query->get_resp_port()->resp(_this->mtxtran_query);
    _this->mtxtran_query = NULL;
    _this->trace.msg("[LightMtxTran] Finished : %0d ns ---> %0d ns | period = %0d ns | runtime = %0d ns | id = %0d\n", start_time_ns, end_time_ns, period_ns, _this->total_runtime, _this->num_transpose);
}
