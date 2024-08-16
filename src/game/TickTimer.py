class EventState:
    ONCE = 0
    FOREVER = 1
    TERMINATE = 2


class Event:
    def __init__(
        self, ticks: int, task, forever: bool, priority: int, info: str = None
    ):
        self.current = int(ticks)
        self.task = task
        self.priority = priority
        self.state = EventState.FOREVER if forever else EventState.ONCE
        self.info = info
        self.ticks = int(ticks)

    def terminate(self):
        self.state = EventState.TERMINATE

    def tick(self):
        if self.current == 0:
            return True
        self.current -= 1
        return False

    def __str__(self) -> str:
        state = "ONCE" if self.state == EventState.ONCE else "FOREVER"
        state = "TERMINATE" if self.state == EventState.TERMINATE else state
        text = f"prior: {self.priority}, ticks: {self.ticks}, state: {state}, info: {self.info}"
        return text


class TickTimer:
    def __init__(self, fps):
        self.fps = fps
        self.events: list[Event] = []

    def add_timer(
        self,
        seconds: int,
        task,
        forever: bool = False,
        priority: int = 0,
        info: str = None,
    ):
        """priority bigger, priority higher"""
        event = Event(int(seconds * self.fps), task, forever, priority, info)
        self.events.append(event)
        self.events.sort(key=lambda x: x.priority)
        return event

    def clear(self):
        self.events.clear()

    def tick(self):
        i = len(self.events) - 1
        while i >= 0:
            event = self.events[i]
            if event.state == EventState.TERMINATE:
                self.events.pop(i)
                i -= 1
                continue
            execute = event.tick()
            if not execute:
                i -= 1
                continue

            event.task()
            if event.state == EventState.ONCE:
                self.events.pop(i)
            if event.state == EventState.FOREVER:
                event.current = event.ticks

    def __str__(self) -> str:
        for event in self.events:
            print(event)
        return ""
