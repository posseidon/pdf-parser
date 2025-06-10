import threading

class ParallelFileLoader:
    def __init__(self, filename, chunk_size=1024):
        self.filename = filename
        self.chunk_size = chunk_size
        self.chunks = []
        self.lock = threading.Lock()

    def _read_chunk(self, start, size):
        try:
            with open(self.filename, 'rb') as f:
                f.seek(start)
                data = f.read(size)
                with self.lock:
                    self.chunks.append(data)
        except FileNotFoundError:
            print(f"File not found: {self.filename}")

    def load_in_chunks(self, num_threads=4):
        try:
            with open(self.filename, 'rb') as f:
                f.seek(0, 2)
                file_size = f.tell()
        except FileNotFoundError:
            print(f"File not found: {self.filename}")
            return []

        threads = []
        for i in range(0, file_size, self.chunk_size):
            t = threading.Thread(target=self._read_chunk, args=(i, self.chunk_size))
            threads.append(t)
            t.start()
            if len(threads) >= num_threads:
                for t in threads:
                    t.join()
                threads = []
        for t in threads:
            t.join()
        return self.chunks
