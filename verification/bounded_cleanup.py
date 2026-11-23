"""Reusable native cleanup with a hard per-search deadline."""


import multiprocessing
import time


from training_data.planar_488.cleanup_decoder import TesseractCleanup


def decode_worker(connection, circuit):
    """Keep the C++ search in a process that can be stopped at a hard deadline."""
    cleanup = TesseractCleanup(circuit)
    connection.send({"ready": True})
    while True:
        syndrome = connection.recv()
        if syndrome is None:
            return
        started = time.perf_counter()
        prediction = cleanup.decode_one(syndrome)
        elapsed = time.perf_counter() - started
        error_indices = list(cleanup.decoder.predicted_errors_buffer)
        witness_cost = cleanup.decoder.cost_from_errors(error_indices)
        connection.send({
            "status": "completed",
            "seconds": elapsed,
            "logical": prediction.logical.tolist(),
            "unreliable": bool(prediction.unreliable),
            "witness_errors": len(error_indices),
            "witness_cost": witness_cost,
        })


class BoundedCleanup:
    """Use one CPU sequentially; restart only after a timed-out search."""

    def __init__(self, circuit, timeout):
        self.circuit = circuit
        self.timeout = timeout
        self.process = None
        self.connection = None

    def close(self):
        if self.process is None:
            return
        if self.process.is_alive():
            self.process.terminate()
        self.process.join()
        self.connection.close()
        self.process = None

    def start(self):
        context = multiprocessing.get_context("fork")
        parent_connection, child_connection = context.Pipe()
        self.connection = parent_connection
        arguments = (child_connection, self.circuit)
        self.process = context.Process(target=decode_worker, args=arguments)
        self.process.start()
        child_connection.close()
        if not self.connection.poll(60):
            self.close()
            raise RuntimeError("Cleanup initialization exceeded 60 seconds")
        self.connection.recv()

    def decode(self, syndrome):
        if self.process is None:
            self.start()
        self.connection.send(syndrome)
        if self.connection.poll(self.timeout):
            return self.connection.recv()
        self.close()
        return {"status": "timeout", "seconds_lower_bound": self.timeout}
