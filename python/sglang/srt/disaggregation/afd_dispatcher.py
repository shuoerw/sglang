"""AFD wire transport via Mooncake-EP.

Real implementation. Replaces the stub.

Architecture:
- One process-global MooncakeAFDTransport per server. Owned by the model
  runner; constructed via init_afd_transport() before model load.
- The expert role runs an HTTP bootstrap server (aiohttp in a daemon thread)
  on host:(ffn_port + 1). Attn ranks POST their handles + receive expert
  handles. After bootstrap, both sides call sync_nvlink_ipc_handles locally.
- Per-layer LayerDispatcher views expose attn_send / attn_recv on the attn
  role and ffn_recv / ffn_send on the expert role. Each call drives one of
  the four AFPD kernels via the existing MooncakeEpBuffer pybind methods.
- N-arena layout on the expert side: one arena per attn source. Source ids
  are assigned at registration time (or claimed via --disaggregation-source-id).

World rank layout used by sync_nvlink_ipc_handles:
    rank 0 .. (num_attn_sources - 1)  : attn ranks (one slot per source)
    rank num_attn_sources             : expert
"""

import contextvars
import dataclasses
import json
import logging
import threading
import time
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------- contextvar ----------------------------

_AFD_CURRENT_SOURCE_ID = contextvars.ContextVar(
    "_AFD_CURRENT_SOURCE_ID", default=None
)


def set_current_source_id(sid):
    return _AFD_CURRENT_SOURCE_ID.set(sid)


def reset_current_source_id(token):
    _AFD_CURRENT_SOURCE_ID.reset(token)


def get_current_source_id():
    return _AFD_CURRENT_SOURCE_ID.get()


# ---------------------------- bootstrap wire ----------------------------

@dataclasses.dataclass
class AttnHandshake:
    source_id: int
    ipc_handle: list
    raddr: int
    rkey: int
    qpns: list
    lids: list
    is_roce: bool
    subnet_prefix: int = 0
    interface_id: int = 0


@dataclasses.dataclass
class ExpertHandshake:
    arena_idx: int
    expert_world_rank: int
    world_size: int
    ipc_handle: list
    raddr: int
    rkey: int
    qpns: list
    lids: list
    is_roce: bool
    subnet_prefix: int = 0
    interface_id: int = 0


# ---------------------------- module-level ----------------------------

_TRANSPORT = None


def init_afd_transport(
    role,
    num_layers,
    num_attn_sources,
    hidden_size,
    top_k,
    num_experts=256,
    num_max_dispatch_tokens_per_rank=1024,
    ffn_addr=None,
    source_id=None,
    use_fp8=True,
):
    global _TRANSPORT
    if _TRANSPORT is not None:
        return _TRANSPORT
    _TRANSPORT = MooncakeAFDTransport(
        role=role,
        num_layers=num_layers,
        num_attn_sources=num_attn_sources,
        hidden_size=hidden_size,
        top_k=top_k,
        num_experts=num_experts,
        num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
        ffn_addr=ffn_addr,
        source_id=source_id,
        use_fp8=use_fp8,
    )
    return _TRANSPORT


def get_transport():
    return _TRANSPORT


def get_or_create_layer_dispatcher(layer_id):
    if _TRANSPORT is None:
        raise RuntimeError(
            "AFD transport not initialized; init_afd_transport must be called "
            "before any AFD-enabled layer is forwarded."
        )
    return _TRANSPORT.get_layer(layer_id)


# ---------------------------- transport ----------------------------

class MooncakeAFDTransport:
    def __init__(
        self,
        role,
        num_layers,
        num_attn_sources,
        hidden_size,
        top_k,
        num_experts,
        num_max_dispatch_tokens_per_rank,
        ffn_addr,
        source_id,
        use_fp8,
    ):
        assert role in ("attn", "ffn"), f"unknown role {role!r}"
        self.role = role
        self.num_layers = num_layers
        self.num_attn_sources = num_attn_sources
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.use_fp8 = use_fp8
        self.world_size = num_attn_sources + 1
        self.expert_world_rank = num_attn_sources

        # Parse ffn_addr (host:port)
        if ffn_addr is None:
            raise ValueError("disaggregation_ffn_addr is required for AFD")
        self.ffn_host, port_str = ffn_addr.split(":")
        self.ffn_port = int(port_str)
        self.bootstrap_port = self.ffn_port + 1

        # Resolve role-specific world rank.
        if role == "attn":
            assert source_id is not None
            self.source_id = source_id
            self.world_rank = source_id
        else:
            self.source_id = None  # ffn doesn't have a source_id
            self.world_rank = self.expert_world_rank

        # Per-expert math.
        self.num_expert_ranks = 1  # M1: single expert rank
        self.num_local_experts = num_experts // self.num_expert_ranks

        # Construct buffer + collect own handles.
        self._construct_buffer()

        # Bootstrap state.
        self._handshake_done = False
        self._handshake_lock = threading.Lock()
        self._registered = {}   # source_id -> AttnHandshake (ffn-side)
        self._reg_lock = threading.Lock()

        if role == "ffn":
            self._start_bootstrap_server()

        # Per-layer dispatcher views.
        self._layers = [LayerDispatcher(self, i) for i in range(num_layers)]
        logger.info(
            "AFD transport initialized: role=%s rank=%d world_size=%d "
            "layers=%d attn_sources=%d num_experts=%d ibgda=%s ffn_addr=%s",
            role, self.world_rank, self.world_size,
            num_layers, num_attn_sources, num_experts,
            not self.buffer.ibgda_disabled(), ffn_addr,
        )

    # --------------------- buffer ---------------------

    def _construct_buffer(self):
        from mooncake import ep
        from mooncake import pg as mooncake_pg
        import torch.distributed as dist

        num_bytes = ep.get_ep_buffer_size_hint(
            self.num_max_dispatch_tokens_per_rank,
            self.hidden_size,
            self.world_size,
            self.num_experts,
        )
        # Need a torch.distributed group for get_preferred_hca. Use WORLD.
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before AFD transport"
            )
        device = f"cuda:{torch.cuda.current_device()}"
        hca = mooncake_pg.get_preferred_hca(dist.group.WORLD, device)
        if not hca:
            raise RuntimeError(
                "get_preferred_hca returned empty; cannot construct ep.Buffer"
            )
        self.buffer = ep.Buffer(self.world_rank, self.world_size, num_bytes, hca)
        logger.info(
            "AFD ep.Buffer constructed: rank=%d world_size=%d num_bytes=%d hca=%s",
            self.world_rank, self.world_size, num_bytes, hca,
        )

        # Collect own handles.
        self._my_ipc = list(self.buffer.get_ipc_handle())
        if self.buffer.ibgda_disabled():
            logger.warning("AFD: IBGDA disabled; using NVLink-IPC only.")
            self._raddr = 0
            self._rkey = 0
            self._qpns = []
            self._lids = []
            self._is_roce = False
            self._subnet = 0
            self._iface = 0
        else:
            mr_addr, mr_rkey = self.buffer.get_mr_info()
            self._raddr = int(mr_addr)
            self._rkey = int(mr_rkey)
            self._qpns = list(self.buffer.get_local_qpns())
            self._lids = list(self.buffer.get_local_lids())
            self._is_roce = bool(self.buffer.is_roce())
            if self._is_roce:
                sub, iface = self.buffer.get_gid()
                self._subnet = int(sub)
                self._iface = int(iface)
            else:
                self._subnet = 0
                self._iface = 0

    # --------------------- bootstrap server (ffn) ---------------------

    def _start_bootstrap_server(self):
        from aiohttp import web
        import asyncio

        self._bootstrap_loop = None

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._bootstrap_loop = loop
            app = web.Application()
            app.router.add_post("/register", self._http_register)
            app.router.add_get("/expert_handshake", self._http_get_expert)
            runner = web.AppRunner(app)
            loop.run_until_complete(runner.setup())
            site = web.TCPSite(runner, self.ffn_host, self.bootstrap_port)
            loop.run_until_complete(site.start())
            logger.info(
                "AFD bootstrap HTTP server listening on %s:%d",
                self.ffn_host, self.bootstrap_port,
            )
            loop.run_forever()

        threading.Thread(target=_run, daemon=True, name="afd-bootstrap").start()

    async def _http_register(self, request):
        from aiohttp import web

        body = await request.json()
        attn = AttnHandshake(**body)
        with self._reg_lock:
            if attn.source_id in self._registered:
                logger.warning(
                    "AFD: source_id=%d re-registering; replacing prior handles",
                    attn.source_id,
                )
            self._registered[attn.source_id] = attn
            n_reg = len(self._registered)
        logger.info(
            "AFD bootstrap: attn source_id=%d registered (total=%d/%d)",
            attn.source_id, n_reg, self.num_attn_sources,
        )
        resp = ExpertHandshake(
            arena_idx=attn.source_id,
            expert_world_rank=self.expert_world_rank,
            world_size=self.world_size,
            ipc_handle=self._my_ipc,
            raddr=self._raddr,
            rkey=self._rkey,
            qpns=self._qpns,
            lids=self._lids,
            is_roce=self._is_roce,
            subnet_prefix=self._subnet,
            interface_id=self._iface,
        )
        return web.json_response(dataclasses.asdict(resp))

    async def _http_get_expert(self, request):
        from aiohttp import web

        return web.json_response({
            "ipc_handle": self._my_ipc,
            "raddr": self._raddr,
            "rkey": self._rkey,
            "qpns": self._qpns,
            "lids": self._lids,
            "is_roce": self._is_roce,
            "subnet_prefix": self._subnet,
            "interface_id": self._iface,
            "n_registered": len(self._registered),
            "num_attn_sources": self.num_attn_sources,
        })

    # --------------------- handshake (lazy) ---------------------

    def ensure_handshake(self):
        with self._handshake_lock:
            if self._handshake_done:
                return
            if self.role == "attn":
                self._do_attn_handshake()
            else:
                self._do_ffn_handshake()
            self._handshake_done = True

    def _do_attn_handshake(self):
        import requests

        url = f"http://{self.ffn_host}:{self.bootstrap_port}"
        body = dataclasses.asdict(AttnHandshake(
            source_id=self.source_id,
            ipc_handle=self._my_ipc,
            raddr=self._raddr, rkey=self._rkey,
            qpns=self._qpns, lids=self._lids,
            is_roce=self._is_roce,
            subnet_prefix=self._subnet, interface_id=self._iface,
        ))
        last_err = None
        for attempt in range(120):
            try:
                r = requests.post(url + "/register", json=body, timeout=5)
                if r.status_code == 200:
                    expert = ExpertHandshake(**r.json())
                    break
            except Exception as e:
                last_err = e
            time.sleep(1)
        else:
            raise RuntimeError(
                f"AFD attn: bootstrap registration failed after 120s: {last_err}"
            )
        logger.info(
            "AFD attn source_id=%d: registered with expert at %s; expert_rank=%d",
            self.source_id, url, expert.expert_world_rank,
        )
        self._sync_local_handles(expert=expert)

    def _do_ffn_handshake(self):
        # Wait for all N attn ranks to register.
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            with self._reg_lock:
                if len(self._registered) >= self.num_attn_sources:
                    break
            time.sleep(0.5)
        else:
            raise RuntimeError(
                f"AFD ffn: only {len(self._registered)}/{self.num_attn_sources} "
                "attn ranks registered after 10 minutes"
            )
        logger.info("AFD ffn: all %d attn ranks registered", self.num_attn_sources)
        self._sync_local_handles(expert=None)

    def _sync_local_handles(self, expert: Optional[ExpertHandshake]):
        """Build the world-rank-ordered handle vectors and call sync_*."""
        # Defaults (own handle as filler for inactive slots).
        remote_ipcs = [self._my_ipc] * self.world_size
        remote_raddrs = [self._raddr] * self.world_size
        remote_rkeys = [self._rkey] * self.world_size
        remote_qpns = [self._qpns] * self.world_size
        remote_lids = [self._lids] * self.world_size
        active_mask = [0] * self.world_size

        if self.role == "attn":
            assert expert is not None
            # active: self + expert (other attn ranks ignored)
            active_mask[self.world_rank] = 1
            active_mask[self.expert_world_rank] = 1
            remote_ipcs[self.expert_world_rank] = list(expert.ipc_handle)
            remote_raddrs[self.expert_world_rank] = int(expert.raddr)
            remote_rkeys[self.expert_world_rank] = int(expert.rkey)
            remote_qpns[self.expert_world_rank] = list(expert.qpns)
            remote_lids[self.expert_world_rank] = list(expert.lids)
        else:
            # active: all attn ranks + self
            active_mask[self.world_rank] = 1
            with self._reg_lock:
                for sid, h in self._registered.items():
                    active_mask[sid] = 1
                    remote_ipcs[sid] = list(h.ipc_handle)
                    remote_raddrs[sid] = int(h.raddr)
                    remote_rkeys[sid] = int(h.rkey)
                    remote_qpns[sid] = list(h.qpns)
                    remote_lids[sid] = list(h.lids)

        try:
            self.buffer.sync_nvlink_ipc_handles(remote_ipcs, active_mask)
            logger.info(
                "AFD %s: sync_nvlink_ipc_handles ok (active_mask=%s)",
                self.role, active_mask,
            )
        except Exception as e:
            logger.error("AFD %s: sync_nvlink_ipc_handles failed: %s", self.role, e)
            raise

        if not self._is_roce and not self.buffer.ibgda_disabled():
            try:
                self.buffer.sync_ib(
                    remote_raddrs, remote_rkeys, remote_qpns, remote_lids, active_mask
                )
                logger.info("AFD %s: sync_ib ok", self.role)
            except Exception as e:
                logger.warning("AFD %s: sync_ib failed: %s", self.role, e)

    # --------------------- per-layer view ---------------------

    def get_layer(self, layer_id):
        return self._layers[layer_id]


# ---------------------------- per-layer dispatcher ----------------------------

class LayerDispatcher:
    def __init__(self, transport: MooncakeAFDTransport, layer_id: int):
        self.t = transport
        self.layer_id = layer_id
        # Attn-side cache: topk_idx/topk_weights captured during attn_send so
        # combine_recv has them.
        self._attn_pending_topk = None  # (topk_idx, topk_weights, num_tokens)

    # --------------------- ATTN role ---------------------

    def attn_send(self, hidden_states, topk_output):
        self.t.ensure_handshake()
        # AFPD dispatch_send: per-expert routed token shipping to expert(s).
        # We fire and forget on this stream; the recv kernel polls signals.
        topk_idx = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
        # afpd kernels expect int64 topk_idx
        if topk_idx.dtype != torch.int64:
            topk_idx = topk_idx.to(torch.int64)
        self.t.buffer.afpd_dispatch_send(
            hidden_states,
            topk_idx,
            self.t.num_max_dispatch_tokens_per_rank,
            self.t.num_experts,
            self.t.world_rank,                # my_attn_rank (world rank)
            self.t.expert_world_rank,         # expert_rank_base
            self.t.num_attn_sources,          # num_attn_ranks
            self.t.num_expert_ranks,          # num_expert_ranks
            self.t.source_id,                 # dst_arena_idx (which arena on expert side)
            self.t.use_fp8,
        )
        self._attn_pending_topk = (topk_idx, topk_weights, hidden_states.shape[0])

    def attn_recv(self):
        topk_idx, topk_weights, num_tokens = self._attn_pending_topk
        self._attn_pending_topk = None
        combined = self.t.buffer.afpd_combine_recv(
            topk_idx,
            topk_weights,
            num_tokens,
            self.t.hidden_size,
            self.t.num_experts,
            self.t.world_rank,                # my_attn_rank
            self.t.num_attn_sources,          # num_attn_ranks
        )
        return combined

    # --------------------- FFN role ---------------------

    def ffn_recv(self):
        self.t.ensure_handshake()
        sid = get_current_source_id()
        if sid is None:
            raise RuntimeError("AFD ffn_recv: no source_id in context")
        recv_x, recv_x_scales, src_info, layout_range, recv_count = \
            self.t.buffer.afpd_dispatch_recv(
                self.t.num_max_dispatch_tokens_per_rank,
                self.t.hidden_size,
                self.t.num_local_experts,
                sid,                          # src_attn_rank
                self.t.num_attn_sources,
                sid,                          # arena_idx
                self.t.use_fp8,
            )
        # Stash per-layer state needed for ffn_send.
        self._ffn_src_info = src_info
        self._ffn_layout_range = layout_range
        self._ffn_source_id = sid

        # Build a DeepEPLLDispatchOutput-shaped object for run_moe_core.
        # topk_ids / topk_weights are unused by run_moe_core's masked path
        # (the routing decision was made on the wire); pass dummy tensors.
        from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLDispatchOutput
        dummy_topk_ids = torch.zeros(
            (1, self.t.top_k), dtype=torch.int64, device=recv_x.device,
        )
        dummy_topk_weights = torch.zeros(
            (1, self.t.top_k), dtype=torch.float32, device=recv_x.device,
        )
        # expected_m: per-expert receive count upper bound
        expected_m = self.t.num_max_dispatch_tokens_per_rank * self.t.num_attn_sources
        return DeepEPLLDispatchOutput(
            hidden_states=recv_x,
            hidden_states_scale=recv_x_scales,
            topk_ids=dummy_topk_ids,
            topk_weights=dummy_topk_weights,
            masked_m=recv_count,
            expected_m=expected_m,
        )

    def ffn_send(self, combine_input):
        # combine_input is a DeepEPLLCombineInput (named tuple): hidden_states, topk_ids, topk_weights
        # We only need hidden_states for combine_send; the topk decision was made on attn side.
        x_local = combine_input.hidden_states
        sid = self._ffn_source_id
        self.t.buffer.afpd_combine_send(
            x_local,
            self._ffn_src_info,
            self._ffn_layout_range,
            self.t.num_max_dispatch_tokens_per_rank,
            self.t.num_local_experts,
            0,                                # my_expert_rank_in_role (M1: only one)
            sid,                              # dst_attn_rank (where to combine back)
            sid,                              # dst_arena_idx
        )
