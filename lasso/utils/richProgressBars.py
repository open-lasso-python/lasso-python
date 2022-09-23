import math
import time
from typing import Any

from rich.progress import ProgressColumn


class PlaceHolderBar():
    finished: bool
    tasks: list = []

    def __init__(self, **kwargs):
        '''This is a placeholder to not clutter console during testing'''
        self.finished: False
    
    def render(self, task: Any) -> str:
        '''returns the planned output: empty string'''
        return ""
    
    def add_task(self, description: str, total: int) -> int:
        '''Adds a new task'''
        self.tasks.append([description, total, 0])
        # entry in list is tupel of description, total tasks, remaining tasks
        return len(self.tasks)-1
    
    def advance(self, task_id):
        '''advanves the given task'''
        prog = self.tasks[task_id][2]
        prog += 1
        self.tasks[task_id][2] = prog
        if prog == self.tasks[task_id][1]:
            self.finished = True
    
    def __enter__(self):
        self.finished = False
    
    def __exit__(self, exception_type, exception_value, traceback):
        self.finished = True
    


class WorkingDots(ProgressColumn):
    max_refresh = 0.5
    is_silenced: bool = False

    def __init__(self, output=True):
        self.counter = 0
        if not output:
            self.is_silenced = True
        super().__init__()

    def render(self, task: Any) -> str:
        self.counter += 1
        if self.is_silenced:
            return ""
        if(task.completed == task.total):
            msg = "..."
        elif(self.counter % 3 == 0):
            msg = ".  "
        elif(self.counter % 3 == 1):
            msg = ".. "
        else:
            msg = "..."
            self.counter = 2
        return msg


class SubsamplingWaitTime(ProgressColumn):
    max_refresh = 0.5

    def __init__(self, n_proc: int):
        super().__init__()

        # Last time we updated
        self.last_time = time.time()
        # Cummulated time of all completed supsampling processes
        self.cum_time = 0
        # Number of parallel running processes
        self.n_proc = n_proc
        self.t_rem = -1

    def render(self, task: Any) -> str:
        """ ?
        """

        if (task.completed == task.total):
            return "Time remaining: 00:00"
        elif self.cum_time > 0:
            avrg_time = self.cum_time / max(1, task.completed)
            rem_tasks = task.total - task.completed
            gr_tasks = math.floor(rem_tasks / self.n_proc)
            if(rem_tasks % self.n_proc) != 0:
                gr_tasks += 1

            total_time = gr_tasks * avrg_time
            if (self.t_rem < 0 or self.t_rem > total_time):
                self.t_rem = total_time

            t_out = self.t_rem - (time.time() - self.last_time)
            mins = str(math.floor(t_out / 60))
            secs = str(math.trunc(t_out % 60))

            if(len(mins) == 1):
                mins = "0" + mins
            if(len(secs) == 1):
                secs = "0" + secs
            out_str = "Time remaining: " + mins + ":" + secs
            return out_str
        else:
            return "Time remaining: --:--"

    def update_avrg(self, new_time: float):
        """ ?
        """

        self.cum_time += new_time
        self.last_time = time.time()
