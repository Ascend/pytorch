import os
import pickle
import select
import time
import multiprocessing

from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.profiler.analysis.prof_common_func._task_manager import (
    ConcurrentTasksManager, send_print_req_to_manager, send_result_to_manager,
    TaskMsgType, ConcurrentTask, ConcurrentMode, TaskStatus
)

from torch_npu.testing.testcase import TestCase, run_tests


class TaskSuccess(ConcurrentTask):

    def __init__(self, deps: list, mode: int):
        self.name = "task_success"
        super().__init__(self.name, deps, mode)

    def run(self, user_input: dict):
        return Constant.SUCCESS, "task_success_output"


class TaskFailed(ConcurrentTask):

    def __init__(self, deps: list, mode: int):
        self.name = "task_fail"
        super().__init__(self.name, deps, mode)

    def run(self, user_input: dict):
        return Constant.FAIL, "task_fail_output"


class TaskException(ConcurrentTask):

    def __init__(self, deps: list, mode: int):
        self.name = "task_exception"
        super().__init__(self.name, deps, mode)

    def run(self, user_input: dict):
        raise RuntimeError("Raise Error!")


class TaskSerial1(ConcurrentTask):

    def __init__(self, deps: list, mode: int):
        self.name = "task_serial1"
        super().__init__(self.name, deps, mode)

    def run(self, user_input: dict):
        return Constant.SUCCESS, "Trans_data"


class TaskSerial2(ConcurrentTask):

    def __init__(self, deps: list, mode: int):
        self.name = "task_serial2"
        super().__init__(self.name, deps, mode)

    def run(self, user_input: dict):
        deps_data = user_input.get("task_serial1")
        if deps_data != "Trans_data":
            raise RuntimeError("Failed to get depend data!")
        return Constant.SUCCESS, "task_serial2_output"


class TestTaskManager(TestCase):

    def setUp(self):
        self.recv_buffer = None
        self.output = None
        self.text = None

    def tearDown(self) -> None:
        pass

    def test_send_print_req_to_manager(self):
        expect_data = "print data"
        self.__send_and_receive_msg(self.__send_print, expect_data)
        self.assertEqual(expect_data, self.text)
        self.output = None
        self.text = None

    def test_send_result_to_manager(self):
        expect_data = {"Name": "ZhangShan"}
        self.__send_and_receive_msg(self.__send_result, expect_data)
        self.assertEqual(expect_data, self.output)
        self.output = None
        self.text = None

    def test_run_in_main_process(self):
        manager = ConcurrentTasksManager()
        task_success = TaskSuccess([], ConcurrentMode.MAIN_PROCESS)
        task_fail = TaskFailed([], ConcurrentMode.MAIN_PROCESS)
        task_exception = TaskException([], ConcurrentMode.MAIN_PROCESS)
        manager.add_task(task_success)
        manager.add_task(task_fail)
        manager.add_task(task_exception)
        manager.run()
        task_infos = manager.task_infos
        self.assertEqual(TaskStatus.Succeed, task_infos.get("task_success").status)
        self.assertEqual(TaskStatus.Failed, task_infos.get("task_fail").status)
        self.assertEqual(TaskStatus.Running, task_infos.get("task_exception").status)

    def test_run_in_sub_process(self):
        manager = ConcurrentTasksManager()
        task_success = TaskSuccess([], ConcurrentMode.SUB_PROCESS)
        task_fail = TaskFailed([], ConcurrentMode.SUB_PROCESS)
        task_exception = TaskException([], ConcurrentMode.SUB_PROCESS)
        manager.add_task(task_success)
        manager.add_task(task_fail)
        manager.add_task(task_exception)
        manager.run()
        task_infos = manager.task_infos
        self.assertEqual(TaskStatus.Succeed, task_infos.get("task_success").status)
        self.assertEqual(TaskStatus.Failed, task_infos.get("task_fail").status)
        self.assertEqual(TaskStatus.Failed, task_infos.get("task_exception").status)

    def test_run_in_sub_thread(self):
        manager = ConcurrentTasksManager()
        task_success = TaskSuccess([], ConcurrentMode.PTHREAD)
        task_fail = TaskFailed([], ConcurrentMode.PTHREAD)
        manager.add_task(task_success)
        manager.add_task(task_fail)
        manager.run()
        task_infos = manager.task_infos
        self.assertEqual(TaskStatus.Succeed, task_infos.get("task_success").status)
        self.assertEqual(TaskStatus.Failed, task_infos.get("task_fail").status)

    def test_run_sub_process_deps(self):
        manager = ConcurrentTasksManager()
        task_serial1 = TaskSerial1([], ConcurrentMode.SUB_PROCESS)
        task_serial2 = TaskSerial2(["task_serial1"], ConcurrentMode.SUB_PROCESS)
        manager.add_task(task_serial1)
        manager.add_task(task_serial2)
        manager.run()
        task_infos = manager.task_infos
        self.assertEqual(TaskStatus.Succeed, task_infos.get("task_serial1").status)
        self.assertEqual(TaskStatus.Succeed, task_infos.get("task_serial2").status)

    def __send_and_receive_msg(self, func, data):
        epoll = select.epoll()
        pr, pw = os.pipe()
        epoll.register(pr, select.EPOLLIN | select.EPOLLET | select.EPOLLERR | select.EPOLLHUP)
        p = multiprocessing.Process(target=func, args=(pw, data))
        p.start()
        waiting = True
        t0 = time.time()
        while waiting:
            events = epoll.poll()
            for fd, event in events:
                if event & select.EPOLLIN:
                    self.__receive_msg(fd)
                    waiting = False
            if time.time() - t0 > 1:
                break

    def __send_print(self, fd, data):
        send_print_req_to_manager(fd, data)

    def __send_result(self, fd, data):
        send_result_to_manager(fd, 1, data)

    def __receive_msg(self, fd):
        try:
            # 每次读64K（Linux上pipe的buffer大小是64K）
            msg = os.read(fd, 64 * 1024)
        except BlockingIOError:
            return

        if self.recv_buffer:
            msg = self.recv_buffer + msg

        rest_len = len(msg)
        rest_msg = msg
        while rest_len > 0:
            # 如果消息不完整，先存到buffer里，等待下一包消息
            # 消息格式：T(4b)L(4b)V(L)
            if rest_len < 8:
                self.recv_buffer = rest_msg
                return
            value_len = int.from_bytes(rest_msg[4:8], "big")
            if rest_len < (8 + value_len):
                self.recv_buffer = rest_msg
                return
            value_type = int.from_bytes(rest_msg[0:4], "big")
            value = rest_msg[8:8 + value_len]

            rest_len -= (8 + value_len)
            rest_msg = rest_msg[8 + value_len:]

            if value_type == TaskMsgType.OUTPUT.value:
                output = pickle.loads(value)
                self.output = output
            elif value_type == TaskMsgType.PRINT.value:
                text = str(value, encoding="utf-8")
                self.text = text

        # 清buffer，需要buffer的场景上面已经return掉了
        self.recv_buffer = None


if __name__ == "__main__":
    run_tests()
