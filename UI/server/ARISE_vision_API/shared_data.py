import threading

# Thread-safe shared data structure
class SharedState:
    def __init__(self):
        self._lock = threading.RLock()  # Reentrant lock for nested locking
        self._data = {}
        self.running = threading.Event()


    def set_value(self, key, value):
        with self._lock:
            self._data[key] = value

    def get_value(self, key):
        with self._lock:
            value = self._data.get(key, None)
            return value

    def delete_value(self, key):
        with self._lock:
            if key in self._data:
                del self._data[key]

    def get_all_data(self):
        with self._lock:
            return self._data.copy()

# Global instance to be shared across threads
shared_state = SharedState()

# Example shared function (can also be thread-safe if needed)
def log_message(msg):
    with threading.Lock():
        print(f"[LOG] {msg}")
