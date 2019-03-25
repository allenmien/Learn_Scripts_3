# -*-coding:utf-8-*-
"""
@Time   : 2019-03-20 11:11
@Author : Mark
@File   : bb.py
"""
import multiprocessing as mp
from queue import Queue

import gevent


class worker(object):
    def __init__(self, Queue):
        self.worker_gevent(Queue)

    def worker_gevent(self, Queue):
        worker_task_list = list()
        for _ in range(10):
            worker_task_list.append(gevent.spawn(self.worker_detail, Queue))
        gevent.joinall(worker_task_list)

    def worker_detail(self, Queue):
        while 1:
            task = Queue.get()
            print('Worker  got task %s' % (task))

            if task is None:
                break
        print('Quitting time!')


class boss(object):
    def __init__(self, Queue):
        self.boss_gevent(Queue)

    def boss_gevent(self, Queue):
        boss_task_list = list()
        for _ in range(10):
            boss_task_list.append(gevent.spawn(self.boss_detail, Queue))
        gevent.joinall(boss_task_list)

    def boss_detail(self, Queue):
        for i in range(1, 25):
            if i == 10:
                gevent.sleep(10)
            Queue.put(i)
        Queue.put(None)


def main():
    processes = list()
    q = Queue()
    for _ in range(4):
        boss_process = mp.Process(target=boss, args=(q,))
        boss_process.start()
        processes.append(boss_process)

    for _ in range(4):
        worker_process = mp.Process(target=worker, args=(q,))
        worker_process.start()
        processes.append(worker_process)

    for process in processes:
        process.join()


if __name__ == '__main__':
    main()
