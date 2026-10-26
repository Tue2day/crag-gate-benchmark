import asyncio
import json

from .io import read_json, write_json


class BudgetExceeded(RuntimeError):
    pass


def estimate_attempt(body, method, cfg):
    # Byte-level token upper estimate plus typed/chat-system overhead. No tiktoken assumption for Jev.
    inputs=len(json.dumps(body,ensure_ascii=False).encode('utf-8'))+1024
    input_price=cfg.input_price_per_million if method=='jev' else cfg.gpt_input_price_per_million
    outputs=0 if method=='jev' else cfg.gpt_max_tokens*cfg.gpt_output_price_per_million
    return (inputs*input_price+outputs)/1e6


class Budget:
    """One process, shared across both models; persist in-flight reservations on every attempt."""
    def __init__(self,path,cfg):
        self.path=path;self.lock=asyncio.Lock()
        if path.exists():
            self.state=read_json(path)
            if self.state['limit_usd']!=cfg.api_budget_usd or self.state['safety_usd']!=cfg.budget_safety_usd:
                raise ValueError('Budget parameters changed; use a new protocol directory')
            # Interrupted attempts may have been billed: never silently release those funds.
            self.state['unknown_reserved_usd']+=sum(self.state['pending'].values())
            self.state['pending']={}
        else:
            self.state={'limit_usd':cfg.api_budget_usd,'safety_usd':cfg.budget_safety_usd,
                'reported_usd':0.0,'unknown_reserved_usd':0.0,'pending':{},'sequence':0,
                'note':'Conservative local budget accounting; unknown attempts retain their estimated charge; verify provider invoice.'}
        write_json(self.path,self.state)

    async def reserve(self,amount):
        async with self.lock:
            committed=self.state['reported_usd']+self.state['unknown_reserved_usd']+sum(self.state['pending'].values())
            if committed+amount>self.state['limit_usd']-self.state['safety_usd']:
                raise BudgetExceeded('Shared API budget exhausted; no new requests; checkpoints retained')
            self.state['sequence']+=1
            ticket=str(self.state['sequence']);self.state['pending'][ticket]=amount
            write_json(self.path,self.state)
            return ticket

    async def settle(self,ticket,cost):
        async with self.lock:
            estimate=self.state['pending'].pop(ticket)
            if cost is None:
                self.state['unknown_reserved_usd']+=estimate
            else:
                self.state['reported_usd']+=cost
            write_json(self.path,self.state)
