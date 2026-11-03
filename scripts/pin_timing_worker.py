"""Pin this experiment worker to one declared Windows logical processor."""
import ctypes as ct
from ctypes import wintypes


def pin_timing_worker(cpu=31):
    mask=1<<cpu
    api=ct.WinDLL('kernel32').SetProcessAffinityMask
    api.argtypes=[wintypes.HANDLE,ct.c_size_t];api.restype=wintypes.BOOL
    if not api(wintypes.HANDLE(-1),mask):raise ct.WinError()
    return mask
