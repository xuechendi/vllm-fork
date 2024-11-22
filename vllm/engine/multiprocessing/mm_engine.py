import pickle
import signal
from typing import List, Optional

import zmq

from vllm.engine.llm_engine import LLMEngine
from vllm.engine.mm_arg_utils import MMAsyncEngineArgs
# yapf conflicts with isort for this block
# yapf: disable
from vllm.engine.multiprocessing import (ENGINE_DEAD_ERROR, IPC_DATA_EXT,
                                         IPC_HEALTH_EXT, IPC_INPUT_EXT,
                                         IPC_OUTPUT_EXT, VLLM_RPC_SUCCESS_STR,
                                         RPCError, RPCProcessRequest)
from vllm.engine.multiprocessing.engine import MQLLMEngine
# yapf: enable
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext

logger = init_logger(__name__)

POLLING_TIMEOUT_MS = 10000
HEALTHY_RESPONSE = (pickle.dumps(VLLM_RPC_SUCCESS_STR), )


class MMLLMEngine(MQLLMEngine):

    def __init__(self,
                 ipc_path: str,
                 use_async_sockets: bool,
                 *args,
                 log_requests: bool = True,
                 **kwargs) -> None:
        # For MQLLMEngine, we can use cached outputs, since each new request
        # output is immediately pickled and send over the socket, which frees
        # the python object to be reused again.
        kwargs['use_cached_outputs'] = True

        # get configs from args and kwargs, determine how many models to load
        original_vllm_config_list = kwargs.get('vllm_config')
        if not isinstance(original_vllm_config_list, List):
            original_vllm_config_list = [original_vllm_config_list]
        self.engines = []

        for i, vllm_config in enumerate(original_vllm_config_list):
            kwargs['vllm_config'] = vllm_config
            self.engines.append(LLMEngine(*args, **kwargs))
        self.engine = self.engines[0]
        self.log_requests = log_requests

        self.use_async_sockets = use_async_sockets
        if self.use_async_sockets:
            for engine in self.engines:
                engine.process_request_outputs_callback = \
                self._async_socket_engine_callback

        self.ctx = zmq.Context()  # type: ignore[attr-defined]

        # Receive input from the client.
        self.input_socket = self.ctx.socket(zmq.constants.PULL)
        self.input_socket.bind(f"{ipc_path}{IPC_INPUT_EXT}")

        # Send output stream back to client.
        self.output_socket = self.ctx.socket(zmq.constants.PUSH)
        self.output_socket.bind(f"{ipc_path}{IPC_OUTPUT_EXT}")

        # Send heartbeats back to client.
        self.heartbeat_socket = self.ctx.socket(zmq.constants.PUSH)
        self.heartbeat_socket.bind(f"{ipc_path}{IPC_HEALTH_EXT}")

        # IPC path for the data socket.
        self.data_ipc_path = f"{ipc_path}{IPC_DATA_EXT}"

        # Error state.
        self._errored_with: Optional[BaseException] = None

    @classmethod
    def from_engine_args(cls, engine_args: MMAsyncEngineArgs,
                         usage_context: UsageContext, ipc_path: str):
        """Creates an MQLLMEngine from the engine arguments."""
        # Setup plugins for each process
        from vllm.plugins import load_general_plugins
        load_general_plugins()

        engine_configs = engine_args.create_engine_configs()
        engine_config = engine_configs[0]
        executor_class = LLMEngine._get_executor_cls(engine_config)

        use_async_sockets = engine_config.model_config.use_async_output_proc

        return cls(ipc_path=ipc_path,
                   use_async_sockets=use_async_sockets,
                   vllm_config=engine_configs,
                   executor_class=executor_class,
                   log_requests=not engine_args.disable_log_requests,
                   log_stats=not engine_args.disable_log_stats,
                   usage_context=usage_context)

    def cleanup(self):
        """Cleanup zeromq state on shutdown."""
        # Closes all sockets and destroys context.
        self.ctx.destroy(linger=0)
        del self.engines

    def run_engine_loop(self):
        """Core busy loop of the LLMEngine."""

        while True:
            if not any(engine.has_unfinished_requests()
                       for engine in self.engines):
                # Poll until there is work to do.
                while self.input_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                    # When there's no work, check on engine health and send
                    # health status back to client
                    self._health_check()
                    for engine in self.engines:
                        engine.do_log_stats()
                    logger.debug("Waiting for new requests in engine loop.")

            # Handle any input from the client.
            self.handle_new_input()

            # Engine step.
            request_outputs = self.engine_step()

            # Send request outputs (if async, done in engine_step callback).
            if not self.use_async_sockets:
                self._send_outputs(request_outputs)

    def engine_step(self) -> List[RequestOutput]:
        """Engine step wrapper with error handling."""
        try:
            res = []
            for engine in self.engines:
                res += engine.step()
            return res
        except SystemExit:
            raise
        except BaseException as e:
            self._set_errored(e)
            rpc_err = RPCError(request_id=None,
                               is_engine_errored=True,
                               exception=e)
            self._send_outputs(rpc_err)
            raise e

    # FIXME: add model field in RPCProcessRequest,
    # and dispatch to the correct engine
    def _handle_process_request(self, request: RPCProcessRequest):
        """Handle RPCProcessRequest by adding it to the LLMEngine."""
        request_id = request.request_id

        if self._errored_with is not None:
            rpc_err = RPCError(request_id=request_id,
                               is_engine_errored=True,
                               exception=ENGINE_DEAD_ERROR(self._errored_with))
            self._send_outputs(rpc_err)

        try:
            for engine in self.engines:
                if engine.model_config.model == request.model:
                    engine.add_request(
                        request_id=request_id,
                        prompt=request.prompt,
                        params=request.params,
                        lora_request=request.lora_request,
                        trace_headers=request.trace_headers,
                        prompt_adapter_request=request.prompt_adapter_request,
                        priority=request.priority)

            if self.log_requests:
                logger.info("Added request %s.", request.request_id)

        except Exception as e:
            # We do not set self._errored = True here, since the error
            # is due to an issue adding this request to the engine,
            # rather than an issue with the engine itself.
            is_errored = self._errored_with is not None
            rpc_err = RPCError(request_id=request_id,
                               is_engine_errored=is_errored,
                               exception=e)
            self._send_outputs(rpc_err)

            # Remove request from the engine.
            self.engine.abort_request(request_id)

    def _health_check(self):
        # Send unhealthy if engine has already errored
        if self._errored_with is not None:
            self._send_unhealthy(self._errored_with)
        try:
            for engine in self.engines:
                engine.check_health()
            self._send_healthy()
        except Exception as e:
            self._set_errored(e)
            self._send_unhealthy(e)


def signal_handler(*_) -> None:
    raise KeyboardInterrupt("MQLLMEngine terminated")


def run_mm_engine(engine_args: MMAsyncEngineArgs, usage_context: UsageContext,
                  ipc_path: str, engine_alive):
    try:
        engine = MMLLMEngine.from_engine_args(engine_args=engine_args,
                                              usage_context=usage_context,
                                              ipc_path=ipc_path)

        signal.signal(signal.SIGTERM, signal_handler)

        engine.start()

    except BaseException as e:
        logger.exception(e)
        engine_alive.value = False
        raise e
