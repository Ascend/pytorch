import os
import sys
import time
import select
import threading
import multiprocessing
import fcntl
import pickle
import signal
import stat
from enum import Enum
from abc import ABC, abstractmethod

from torch_npu.utils._error_code import ErrCode, prof_error
from ._constant import print_error_msg, Constant

__all__ = []


# 并发模式，非互斥的属性可以通过|同时设置
class ConcurrentMode:
    MAIN_PROCESS = 0  # 在主进程中执行，阻塞式，建议用于公共前置任务或好事很短的任务
    SUB_PROCESS = 1  # 独立子进程，建议计算量大、需要独占Cpu核的任务使用
    PTHREAD = 2  # 子线程调度，建议计算量小或是返回数据很大的任务使用
    NON_BLOCKING = 4  # 非阻塞任务，若任务管理器中仅剩具有该属性的任务，则结束调度；建议用于可能卡死的辅助型中间任务


class ConcurrentTask(ABC):
    def __init__(self, name: str, deps: list, mode: int):
        self.name = name
        self.deps = deps
        self.mode = mode

    @property
    def is_non_blocking(self):
        return (self.mode & ConcurrentMode.NON_BLOCKING) != 0

    @abstractmethod
    def run(self, user_input: dict):
        """An abstract method that user must implement.
        :param user_input: Constructed by the framework, based on return of prerequisite tasks,
                      format as {"task_name_1": output1, "task_name_2": output2, ...}
        :return: Except 2 return value, the first is return code(0 for OK and others for error),
                 the second is output.
        """
        raise NotImplementedError("Function run need to be implemented." + prof_error(ErrCode.NOT_SUPPORT))


# 消息以TLV格式发送，TL各4字节
class TaskMsgType(Enum):
    RET_CODE = 1
    OUTPUT = 2
    PRINT = 3


def send_result_to_manager(fd, ret_code, output):
    if fd < 0:
        raise OSError("[Errno 9] Bad file descriptor" + prof_error(ErrCode.UNAVAIL))

    msg = b''
    # 先发output， 再发ret_code， 接受端会在收到ret_code时认为任务执行完成
    if output:
        output_serialized = pickle.dumps(output)
        msg += TaskMsgType.OUTPUT.value.to_bytes(4, "big")
        msg += len(output_serialized).to_bytes(4, "big")
        msg += output_serialized

    msg += TaskMsgType.RET_CODE.value.to_bytes(4, "big")
    msg += (4).to_bytes(4, "big")
    msg += ret_code.to_bytes(4, "big")

    msg_len = len(msg)
    # 预期大多数情况一次就写完了，因此while外面单独写一次减少无用操作
    send_len = os.write(fd, msg)
    while send_len < msg_len:
        msg = msg[send_len:]
        msg_len = msg_len - send_len
        send_len = os.write(fd, msg)


def send_print_req_to_manager(fd, text):
    info = bytes(text, encoding="utf-8")
    msg = TaskMsgType.PRINT.value.to_bytes(4, "big")
    msg += len(info).to_bytes(4, "big")
    msg += info

    msg_len = len(msg)
    send_len = os.write(fd, msg)
    while send_len < msg_len:
        msg = msg[send_len:]
        msg_len = msg_len - send_len
        send_len = os.write(fd, msg)


class TaskStatus(Enum):
    Init = 0
    Running = 1
    Succeed = 2
    Failed = 3
    Stopped = 4


class TaskInfo:
    def __init__(self, task):
        self.task = task
        self.pre_tasks = set(task.deps)
        self.post_tasks = set()
        self.status = TaskStatus.Init
        self.output = None
        self.handler = None
        self.pipe = (-1, -1)
        self.recv_buffer = None


class ConcurrentTasksManager:
    """A concurrent-task manager.
       Create tasks of class ConcurrentTask, add them into manager, then call manager.run().
    """

    def __init__(self, *, max_concurrent_num=16, progress_bar=None):
        self.task_infos = {}  # format: {task_name: task_info, ...}
        self.listening_infos = {}  # format: {recv_fd: task_info, ...}
        self.ready_tasks = []
        self.epoll = None
        self.max_concurrent_num = max_concurrent_num
        self.progress_bar = progress_bar

    def add_task(self, task):
        if not isinstance(task, ConcurrentTask):
            raise TypeError("Task should be an instance of ConcurrentTask" + prof_error(ErrCode.TYPE))
        for pre_task_name in task.deps:
            pre_task_info = self.task_infos.get(pre_task_name)
            if not pre_task_info:
                raise ValueError("Unknow task %s in deps." % pre_task_name + prof_error(ErrCode.VALUE))
            pre_task_info.post_tasks.add(task.name)
        task_info = TaskInfo(task)
        self.task_infos[task.name] = task_info
        if not task.deps:
            self.ready_tasks.append(task_info)

    def run(self):
        try:
            signal.signal(signal.SIGINT, self.finalize)
            if self.progress_bar:
                self.__start_print_progress_bar()

            self.__schedule()
            while True:
                need_exit = self.__listen()
                if need_exit:
                    break
                self.__schedule()
        except Exception as e:
            print_error_msg(f"An error occurred: {e}")
        finally:
            self.finalize()

    def finalize(self):
        for task_info in self.task_infos.values():
            if task_info.status != TaskStatus.Succeed:
                print_error_msg(f"Task [{task_info.task.__class__.__name__}] run failed.")
                self.__stop_task(task_info)

        if self.progress_bar:
            self.__stop_print_progress_bar()

    def clear(self):
        for task_info in self.listening_infos.values():
            self.__stop_task(task_info)
        self.task_infos = {}
        self.listening_infos = {}
        self.ready_tasks = []
        if self.epoll is not None:
            self.epoll.close()
            self.epoll = None

    def __schedule(self):
        """ schedule tasks those are ready """
        free_channel = self.max_concurrent_num - len(self.listening_infos)
        num_to_run = min(free_channel, len(self.ready_tasks))
        tasks_wait_schedule = self.ready_tasks[num_to_run:]
        for task_info in self.ready_tasks[:num_to_run]:
            self.__run_one_task(task_info)
        self.ready_tasks = tasks_wait_schedule

    def __run_one_task(self, task_info):
        task_info.status = TaskStatus.Running
        if (task_info.task.mode & ConcurrentMode.SUB_PROCESS) != 0:
            self.__run_in_subprocess(task_info)
        elif (task_info.task.mode & ConcurrentMode.PTHREAD) != 0:
            self.__run_in_pthread(task_info)
        else:
            self.__run_in_mainprocess(task_info)

    def __run_in_subprocess(self, task_info):
        user_input = {}
        for dep in task_info.task.deps:
            dep_task = self.task_infos.get(dep)
            if dep_task:
                user_input[dep] = self.task_infos.get(dep).output

        def func_nop(*args, **kws):
            pass

        def stdout_wrapper(text):
            send_print_req_to_manager(task_info.pipe[1], text)

        def process_task_func(info, deps_input):
            os.close(info.pipe[0])
            task = info.task
            # 父进程内有其他python线程，此处子进程对stdout操作可能导致死锁，因此打印信息统一发送给父进程处理
            sys.stdout.write = stdout_wrapper
            sys.stdout.flush = func_nop
            ret_code, output = task.run(deps_input)
            if ret_code != 0:
                output = None
            send_result_to_manager(info.pipe[1], ret_code, output)

        self.__add_listening(task_info)
        p = multiprocessing.Process(target=process_task_func, args=(task_info, user_input))
        task_info.handler = p
        p.start()
        os.close(task_info.pipe[1])
        task_info.pipe = (task_info.pipe[0], -1)

    def __run_in_pthread(self, task_info):
        user_input = {}
        for dep in task_info.task.deps:
            dep_task = self.task_infos.get(dep)
            if dep_task:
                user_input[dep] = self.task_infos.get(dep).output

        def thread_task_func(info, deps_input):
            task = info.task
            ret_code, output = task.run(deps_input)
            # 子线程模式与主线程共用地址空间，考虑到output可能很大，此处不用pipe传数据，直接赋值
            # 由于调度关系天然限制了读写时序，此处无需线程锁
            if ret_code == 0:
                info.output = output
            pipe_write = info.pipe[1]
            send_result_to_manager(pipe_write, ret_code, None)
            # 线程模式需要线程内自行关闭写端pipe以触发exit
            info.pipe = (info.pipe[0], -1)
            os.close(pipe_write)

        self.__add_listening(task_info)
        t = threading.Thread(target=thread_task_func, args=(task_info, user_input))
        task_info.handler = t
        t.start()

    def __run_in_mainprocess(self, task_info):
        user_input = {}
        for dep in task_info.task.deps:
            dep_task = self.task_infos.get(dep)
            if dep_task:
                user_input[dep] = self.task_infos.get(dep).output
        ret_code, output = task_info.task.run(user_input)
        self.__on_task_done(task_info, ret_code, output)

    def __on_task_done(self, task_info, ret_code, output):
        """ be called when task.run is finish(listening thread receives ret_code) """
        if ret_code == 0:
            task_info.status = TaskStatus.Succeed
            if output is not None:
                task_info.output = output
            for task_name in task_info.post_tasks:
                post_task = self.task_infos.get(task_name)
                post_task.pre_tasks.remove(task_info.task.name)
                if len(post_task.pre_tasks) == 0:
                    # if all deps have done, the task is ready to run
                    self.ready_tasks.append(post_task)
            for task_name in task_info.task.deps:
                pre_task = self.task_infos.get(task_name)
                pre_task.post_tasks.remove(task_info.task.name)
                if len(pre_task.post_tasks) == 0:
                    # if no task needs the output, clean it for saving memory
                    pre_task.output = None
        else:
            task_info.status = TaskStatus.Failed

    def __on_task_exit(self, task_info):
        """ be called when subprocess/pthread exits """
        # if a task exits without calling __on_task_done, infer that an error occurred
        if task_info.status != TaskStatus.Succeed:
            task_info.status = TaskStatus.Failed
        self.__remove_listening(task_info)

    def __add_listening(self, task_info):
        if self.epoll is None:
            self.epoll = select.epoll()
        pr, pipe_write = os.pipe()

        try:
            # 设置读管道为非阻塞并限制权限
            flags = fcntl.fcntl(pr, fcntl.F_GETFL)
            fcntl.fcntl(pr, fcntl.F_SETFL, flags | os.O_NONBLOCK)

            # 设置管道文件描述符权限（只允许当前用户访问）
            os.fchmod(pr, stat.S_IRUSR | stat.S_IWUSR)
            os.fchmod(pipe_write, stat.S_IRUSR | stat.S_IWUSR)
        except (OSError, AttributeError):
            flags = fcntl.fcntl(pr, fcntl.F_GETFL)
            fcntl.fcntl(pr, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        task_info.pipe = (pr, pipe_write)
        self.epoll.register(pr, select.EPOLLIN | select.EPOLLET | select.EPOLLERR | select.EPOLLHUP)
        self.listening_infos[pr] = task_info

    def __remove_listening(self, task_info):
        pr, pipe_write = task_info.pipe
        if pr != -1:
            self.listening_infos.pop(pr)
            self.epoll.unregister(pr)
            os.close(pr)
        if pipe_write != -1:
            os.close(pipe_write)
        task_info.pipe = (-1, -1)

    def __listen(self):
        # 先检查是否可以退出，若当前无监听任务或监听的都是NON_BLOCKING型任务，则退出
        need_exit = True
        for task_info in self.listening_infos.values():
            if (task_info.task.mode & ConcurrentMode.NON_BLOCKING) == 0:
                need_exit = False
                break
        if need_exit:
            time.sleep(Constant.SLEEP_TIME * 5)
            if all((task_info.task.is_non_blocking for task_info in self.listening_infos.values())):
                return True

        events = self.epoll.poll()
        for fd, event in events:
            if event & select.EPOLLIN:
                self.__on_recv_msg(fd)
            if event & select.EPOLLERR or event & select.EPOLLHUP:
                self.__on_task_exit(self.listening_infos.get(fd))

        return False

    def __on_recv_msg(self, fd):
        try:
            # 每次读64K（Linux上pipe的buffer大小是64K）
            msg = os.read(fd, 64 * 1024)
        except BlockingIOError:
            return
        task_info = self.listening_infos.get(fd)
        if task_info.recv_buffer:
            msg = task_info.recv_buffer + msg

        rest_len = len(msg)
        rest_msg = msg
        while rest_len > 0:
            # 如果消息不完整，先存到buffer里，等待下一包消息
            # 消息格式：T(4b)L(4b)V(L)
            if rest_len < 8:
                task_info.recv_buffer = rest_msg
                return
            value_len = int.from_bytes(rest_msg[4:8], "big")
            if rest_len < (8 + value_len):
                task_info.recv_buffer = rest_msg
                return
            value_type = int.from_bytes(rest_msg[0:4], "big")
            value = rest_msg[8:8 + value_len]

            rest_len -= (8 + value_len)
            rest_msg = rest_msg[8 + value_len:]

            if value_type == TaskMsgType.RET_CODE.value:
                ret_code = int.from_bytes(value, "big")
                self.__on_task_done(task_info, ret_code, None)
            elif value_type == TaskMsgType.OUTPUT.value:
                output = pickle.loads(value)
                task_info.output = output
            elif value_type == TaskMsgType.PRINT.value:
                text = str(value, encoding="utf-8")
                print(text, end='')

        # 清buffer，需要buffer的场景上面已经return掉了
        task_info.recv_buffer = None

    def __stop_task(self, task_info):
        if task_info.handler is None:
            return
        if not task_info.handler.is_alive():
            return

        if (task_info.task.mode & ConcurrentMode.SUB_PROCESS) != 0:
            task_info.handler.terminate()
        elif (task_info.task.mode & ConcurrentMode.PTHREAD) != 0:
            # 线程对象没有直接提供stop方法，此处给该线程的运行栈注入一个中断异常
            import ctypes
            tid = ctypes.c_long(task_info.handler.ident)
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(InterruptedError))

        task_info.status = TaskStatus.Stopped

    def __start_print_progress_bar(self):
        def print_dot_thread_func(m):
            while m.progress_bar:
                print(".", end='', flush=True)
                time.sleep(1)

        def rotate_cursor_thread_func(m):
            s = "|/-\\"
            i = 0
            total = len(s)
            while m.progress_bar:
                print(s[i] + '\b', end='', flush=True)
                time.sleep(1)
                i = (i + 1) % total

        if self.progress_bar == "dot":
            t = threading.Thread(target=print_dot_thread_func, args=(self,))
        elif self.progress_bar == "cursor":
            t = threading.Thread(target=rotate_cursor_thread_func, args=(self,))
        else:
            return
        t.start()

    def __stop_print_progress_bar(self):
        self.progress_bar = None

    def __del__(self):
        self.clear()
