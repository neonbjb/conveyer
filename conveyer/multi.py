from multiprocessing import Process, Queue, Event
import multiprocessing
from typing import Callable, Optional

from conveyer.core import Stage


class Worker:
    def __init__(self, id: int, mapping_fn: Callable, task_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue,
                 shutdown_flag: multiprocessing.Event):
        self.id = id
        self.mapping_fn = mapping_fn
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.shutdown_flag = shutdown_flag

    def __call__(self):
        while not self.shutdown_flag.is_set():
            task = self.task_queue.get()
            if task is None:  # Sentinel value for shutdown
                break
            result = self.mapping_fn(task)
            self.result_queue.put(result)


class DistributedBatchedMap(Stage):
    """
    Allows performing an arbitrary mapping function to the outputs of a pipeline stage, where the mapping_fn is
    invoked in a separate process.
    mapping_fn is a function that will be called for every input on a remote worker. must be picklable.
    n_workers is the number of sub-processes to spawn per GPU
    buffer_size is the number of elements to pre-process at any time
    timeout is the amount of time to wait for an element to be returned. If None, will wait indefinitely.
    """
    def __init__(self, mapping_fn: Callable, n_workers: int, buffer_size: Optional[int] = None, timeout: float = 60 * 60):
        self.inner_state = None
        self.mapping_fn = mapping_fn
        self.n_workers = n_workers
        if buffer_size is None:
            buffer_size = 4 * n_workers
        self.buffer_size = buffer_size
        self.shutdown_flag = multiprocessing.Event()
        self.initialized = False
        self.timeout = timeout
        self._input_stage = None  # Set by Conveyer class.
        self._next_output_index = 0
        self._next_input_index = 0
        self._input_queues = [Queue() for _ in range(self.n_workers)]
        self._result_queues = [Queue() for _ in range(self.n_workers)]

    def _init_workers(self):
        self._worker_processes = []
        for i in range(self.n_workers):
            worker = Worker(id=i, mapping_fn=self.mapping_fn, task_queue=self._input_queues[i],
                                              result_queue=self._result_queues[i], shutdown_flag=self.shutdown_flag)
            p = Process(target=worker)
            p.start()
            self._worker_processes.append(p)
        self.queued_states = []
        self.current_state = []
        self.n_in_queue = 0

    def _enqueue(self, work):
        self._input_queues[self._next_input_index].put(work)
        self.n_in_queue += 1
        self._next_input_index = (self._next_input_index + 1) % self.n_workers
        self.queued_states.append(self._input_stage.state())

    def __call__(self, *args, **kwargs):
        if not self.initialized:
            self._init_workers()
            self.initialized = True
            self.current_state = self._input_stage.state()
            # plumb the pipes
            for _ in range(self.buffer_size):
                try:
                    work = self._input_stage()
                    # Record inner states so that we can restore the last state that was iterated on, rather then the last
                    # buffered state.
                    self._enqueue(work)
                except StopIteration:
                    # Don't have enough data to saturate the buffer; that's OK.
                    break

        try:
            work = self._input_stage()
            self._enqueue(work)
        except StopIteration:
            if self.n_in_queue <= 0:
                raise

        self.current_state = self.queued_states.pop(0)
        self.n_in_queue -= 1
        res = self._result_queues[self._next_output_index].get(timeout=self.timeout)
        self._next_output_index = (self._next_output_index + 1) % self.n_workers
        return res

    def state(self):
        return self.current_state

    def set_state(self, state):
        self.current_state = None
        if self.initialized:
            self.close()
            self.initialized = False
        self._input_stage.set_state(state)

    def close(self):
        self.shutdown_flag.set()
        for i in range(self.n_workers):
            self._input_queues[i].put(None)
        for worker in self._worker_processes:
            worker.join()
