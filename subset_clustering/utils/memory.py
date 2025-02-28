import psutil

def get_CPU_memory():
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total
    available_memory = memory_info.available
    used_memory = memory_info.used
    memory_percent = memory_info.percent
    print('.........CPU Memory.........')
    print('Total CPU memory: %.2f MB' % (total_memory / 
                                         1024 / 1024))
    print('Current CPU memory usage: %.2f MB' % (used_memory /
                                                 1024 / 1024))
    print('Available CPU memory: %.2f MB' % (available_memory / 
                                             1024 / 1024))
