# Simple script to filter some log messages
import sys
import re

class FilterPrint:
    def __init__(self):
        self.p_objs = []
        # To keep block construction statements, remove
        # 'actor_pool_map_operator' from the list below.
        self.p_exclude = [r'streaming_executor', r'actor_pool_map_operator',
                          r'real_accelerator']
        # Comment this out to see data loading stats
        self.p_exclude.extend([r'object_store', r'Running', r'^\n$'])
        self.p_exclude_objs = [re.compile(p) for p in self.p_exclude]

    def has_duplicate_emptylines(self, line):
        try:
            self.__prev_line
        except AttributeError:
            # First line
            self.__prev_line = line
            return False

        is_empty_line_prev = len(self.__prev_line.strip())
        is_empty_line_curr = len(line.strip())
        self.__prev_line = line
        return (is_empty_line_prev and is_empty_line_curr)

    def has_exclude_patterns(self, line):
        for p in self.p_exclude_objs:
            if p.search(line) is not None:
                return True
        return False

    def __call__(self, line):
        exclude_line = self.has_exclude_patterns(line)
        # empty_lines = self.has_duplicate_emptylines(line)
        empty_lines = False
        if exclude_line or empty_lines:
            return
        print(line)


if __name__ == '__main__':
    fprint = FilterPrint()
    try:
        buff = ''
        while True:
            buff += sys.stdin.read(1)
            if buff.endswith('\n'):
                line = buff[:-1]
                fprint(line)
                buff = ''
    except KeyboardInterrupt:
        sys.stdout.flush()
