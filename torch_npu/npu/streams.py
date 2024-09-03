import ctypes

import torch_npu
import torch_npu._C


class Stream(torch_npu._C._NPUStreamBase):
    r"""Wrapper around a NPU stream.

    A NPU stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams.  See :ref:`npu-semantics` for
    details.

    Arguments:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream. Lower numbers
                                 represent higher priorities.
    """

    def __new__(cls, device=None, priority=0, **kwargs):
        with torch_npu.npu.device(device):
            return super(Stream, cls).__new__(cls, priority=priority, **kwargs)

    def wait_event(self, event):
        r"""Makes all future work submitted to the stream wait for an event.

        Arguments:
            event (Event): an event to wait for.

        .. note:: This is a wrapper around ``npuStreamWaitEvent()``

           This function returns without waiting for :attr:`event`: only future
           operations are affected.

        """
        event.wait(self)

    def wait_stream(self, stream):
        r"""Synchronizes with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Arguments:
            stream (Stream): a stream to synchronize.

        .. note:: This function returns without waiting for currently enqueued
           kernels in :attr:`stream`: only future operations are affected.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        r"""Records an event.

        Arguments:
            event (Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self):
        r"""Checks if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed.
        """
        return super(Stream, self).query()

    def synchronize(self):
        r"""Wait for all the kernels in this stream to complete.

        .. note:: This is a wrapper around ``npuStreamSynchronize()``: see
           `NPU Stream documentation`_ for more info.
        """
        super(Stream, self).synchronize()

    def set_data_preprocess_stream(self, is_data_preprocess_stream=False):
        r"""Set data preprocess mode property to this stream.

        Arguments:
            is_data_preprocess_stream(bool): determine
            whether to add data preprocess property.
        """
        super(Stream, self).set_data_preprocess_stream(is_data_preprocess_stream)

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.npu_stream)

    def __eq__(self, other):
        if isinstance(other, Stream):
            return super(Stream, self).__eq__(other)
        return False

    def __hash__(self):
        return hash((self.npu_stream, self.device))

    def __repr__(self):
        return ('<torch_npu.npu.Stream device={0} npu_stream={1:#x}>'
                .format(self.device, self.npu_stream))


class Event(torch_npu._C._NPUEventBase):
    r"""Wrapper around a NPU event.

    NPU events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize NPU
    streams.

    The underlying NPU events are lazily initialized when the event is first
    recorded or exported to another process. After creation, only streams on the
    same device may record the event. However, streams on any device can wait on
    the event.

    Arguments:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)

    """

    def __new__(cls, enable_timing=False, blocking=False, interprocess=False):
        return super(Event, cls).__new__(cls, enable_timing=enable_timing, blocking=blocking, interprocess=interprocess)

    def record(self, stream=None):
        r"""Records the event in a given stream.

        Uses ``torch_npu.npu.current_stream()`` if no stream is specified. The
        stream's device must match the event's device.
        """
        if stream is None:
            stream = torch_npu.npu.current_stream()
        super(Event, self).record(stream)

    def wait(self, stream=None):
        r"""Makes all future work submitted to the given stream wait for this
        event.

        Use ``torch_npu.npu.current_stream()`` if no stream is specified.
        """
        if stream is None:
            stream = torch_npu.npu.current_stream()
        super(Event, self).wait(stream)

    def query(self):
        r"""Checks if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return super(Event, self).query()

    def elapsed_time(self, end_event):
        r"""Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.
        """
        return super(Event, self).elapsed_time(end_event)

    def synchronize(self):
        r"""Waits for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.

         .. note:: This is a wrapper around ``npuEventSynchronize()``: see
            `NPU Event documentation`_ for more info.
        """
        super(Event, self).synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.npu_event)

    def __repr__(self):
        if self.npu_event:
            return '<torch_npu.npu.Event {0:#x}>'.format(self._as_parameter_.value)
        else:
            return '<torch_npu.npu.Event uninitialized>'


class SyncLaunchStream(torch_npu._C._NPUStreamBase):
    r"""Wrapper around a SyncLaunch NPU stream.

    A Sync Launch NPU stream is a NPU stream which doesn't enable taskqueue(implemented by is_sync_launch).

    Arguments:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream. Lower numbers
                                 represent higher priorities.
    """

    def __new__(cls, device=None, priority=0, **kwargs):
        with torch_npu.npu.device(device):
            return super(SyncLaunchStream, cls).__new__(cls, priority=priority, is_sync_launch=1, **kwargs)

    def wait_event(self, event):
        r"""Makes all future work submitted to the stream wait for an event.

        Arguments:
            event (Event): an event to wait for.

        .. note:: This is a wrapper around ``npuStreamWaitEvent()``

           This function returns without waiting for :attr:`event`: only future
           operations are affected.

        """
        event.wait(self)

    def wait_stream(self, stream):
        r"""Synchronizes with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Arguments:
            stream (SyncLaunchStream): a stream to synchronize.

        .. note:: This function returns without waiting for currently enqueued
           kernels in :attr:`stream`: only future operations are affected.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        r"""Records an event.

        Arguments:
            event (Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self):
        r"""Checks if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed.
        """
        return super(SyncLaunchStream, self).query()

    def synchronize(self):
        r"""Wait for all the kernels in this stream to complete.

        .. note:: This is a wrapper around ``npuStreamSynchronize()``: see
           `NPU Stream documentation`_ for more info.
        """
        super(SyncLaunchStream, self).synchronize()

    def set_data_preprocess_stream(self, is_data_preprocess_stream=False):
        r"""Set data preprocess mode property to this stream.

        Arguments:
            is_data_preprocess_stream(bool): determine
            whether to add data preprocess property.
        """
        super(SyncLaunchStream, self).set_data_preprocess_stream(is_data_preprocess_stream)

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.npu_stream)

    def __eq__(self, other):
        if isinstance(other, SyncLaunchStream):
            return super(SyncLaunchStream, self).__eq__(other)
        return False

    def __hash__(self):
        return hash((self.npu_stream, self.device))

    def __repr__(self):
        return ('<torch_npu.npu.SyncLaunchStream device={0} npu_stream={1:#x}>'
                .format(self.device, self.npu_stream))
