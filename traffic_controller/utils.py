import time


def format_uptime(seconds):
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


def print_header(title):
    print()
    print("=" * 60)
    print(title.center(60))
    print("=" * 60)


def print_dict(data, indent=2):
    spaces = " " * indent
    for key, value in data.items():
        print(f"{spaces}{key}: {value}")


class Timer:
    
    def __init__(self, interval):
        self.interval = interval
        self.last_time = time.time()
    
    def check(self):
        current_time = time.time()
        if current_time - self.last_time >= self.interval:
            self.last_time = current_time
            return True
        return False
    
    def reset(self):
        self.last_time = time.time()
    
    def elapsed(self):
        return time.time() - self.last_time