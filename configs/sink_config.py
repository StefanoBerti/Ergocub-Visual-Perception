from logging import INFO

from utils.concurrency import YarpNode
from utils.confort import BaseConfig


class Logging(BaseConfig):
    level = INFO


class Network(BaseConfig):
    node = YarpNode

    class Args:
        in_queue = 'sink'
        out_queues = []

